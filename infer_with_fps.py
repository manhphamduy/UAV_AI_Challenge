import cv2
import torch
import time
import torchvision
import numpy as np
from collections import deque
from tqdm import tqdm
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ==== CONFIG ====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 12
model_path = "models/c2a_best_model.pth" # Sử dụng model tốt nhất
input_video = "test_video.mp4"
output_video = "output_fps_benchmark.avi"
threshold = 0.5

# ==== Load model ====
print("Loading model...")
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()
print("✅ Model loaded successfully")

# ======================================================================
# WARM-UP GIAI ĐOẠN (Rất quan trọng để có kết quả đo chính xác)
# ======================================================================
if device.type == 'cuda':
    print("🚀 Warming up the GPU...")
    # Tạo một tensor giả có kích thước tương tự ảnh đầu vào
    dummy_input = torch.randn(1, 3, 480, 640, device=device)
    for _ in range(10):
        with torch.no_grad():
            model(dummy_input)
    # Đảm bảo tất cả các tác vụ warm-up đã hoàn thành
    torch.cuda.synchronize()
    print("✅ GPU is warm.")

# ==== Load video ====
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    raise RuntimeError(f"❌ Could not open video {input_video}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# Biến để tính toán hiệu suất
model_latencies = []
end_to_end_times = []
# Sử dụng deque để tính FPS trung bình trượt (moving average) cho mượt
fps_smoother = deque(maxlen=30) 

print("🚀 Running inference and benchmarking FPS...")
progress_bar = tqdm(total=total_frames, desc="Processing video")

while True:
    loop_start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # --- Pre-processing ---
    img_tensor = torchvision.transforms.functional.to_tensor(frame).to(device)

    # --- Model Inference (Đo lường chính xác) ---
    torch.cuda.synchronize() # Đảm bảo các tác vụ trước đó đã xong
    inference_start = time.time()
    
    with torch.no_grad():
        pred = model([img_tensor])[0]
        
    # ====> ĐÂY LÀ DÒNG QUAN TRỌNG NHẤT <====
    torch.cuda.synchronize() # Buộc CPU đợi GPU hoàn thành
    inference_end = time.time()
    
    model_latency = inference_end - inference_start
    model_latencies.append(model_latency)

    # --- Post-processing ---
    boxes = pred["boxes"].cpu().numpy()
    scores = pred["scores"].cpu().numpy()
    labels = pred["labels"].cpu().numpy()

    for box, score, label in zip(boxes, scores, labels):
        if score < threshold:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.putText(frame, f"Obj {label.item()} {score:.2f}", ... ) # (Bạn có thể thêm lại nếu muốn)

    # --- Hiển thị FPS (đã làm mượt) ---
    loop_end_time = time.time()
    end_to_end_time = loop_end_time - loop_start_time
    end_to_end_times.append(end_to_end_time)
    
    # Tính FPS trung bình của 30 frame gần nhất
    fps_smoother.append(end_to_end_time)
    smooth_fps = 1.0 / np.mean(fps_smoother)

    cv2.putText(frame, f"FPS: {smooth_fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    out.write(frame)
    progress_bar.update(1)

# --- Dọn dẹp và In kết quả ---
cap.release()
out.release()
progress_bar.close()

# Tính toán các chỉ số trung bình
avg_model_latency_ms = np.mean(model_latencies) * 1000
avg_model_fps = 1.0 / np.mean(model_latencies)
avg_end_to_end_fps = 1.0 / np.mean(end_to_end_times)

print(f"\n✅ Done! Video saved to {output_video}")
print("="*30)
print("📊 BENCHMARK RESULTS 📊")
print("="*30)
print(f"⏱️ Model Inference Latency: {avg_model_latency_ms:.2f} ms/frame")
print(f"🚀 Model Throughput:        {avg_model_fps:.2f} FPS")
print(f"🎬 End-to-End Throughput:   {avg_end_to_end_fps:.2f} FPS")
print("="*30)