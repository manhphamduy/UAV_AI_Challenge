import cv2
import torch
import torchvision
import argparse
from tqdm import tqdm
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# --- CONFIG ---
CLASSES = [
    '__background__', 'pedestrian', 'person', 'car', 'van', 'bus', 
    'truck', 'motor', 'bicycle', 'awning-tricycle', 'tricycle', 'other'
]

COLORS = torch.randint(0, 255, (len(CLASSES), 3))

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ==== Load model ====
    print("Loading model...")
    num_classes = len(CLASSES)
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    print("✅ Model loaded for inference")

    # ==== Process video ====
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {args.video_path}")

    # Lấy thông tin video để tạo output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (frame_width, frame_height))

    print(f"Processing video: {args.video_path}")
    progress_bar = tqdm(total=total_frames, desc="Processing frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Chuyển đổi frame sang tensor
        img_tensor = torchvision.transforms.functional.to_tensor(frame).to(device)

        with torch.no_grad():
            pred = model([img_tensor])[0]

        # Lấy kết quả và chuyển về CPU
        boxes = pred["boxes"].cpu()
        scores = pred["scores"].cpu()
        labels = pred["labels"].cpu()

        # Vẽ kết quả lên frame
        for box, score, label_idx in zip(boxes, scores, labels):
            if score < args.threshold:
                continue
            
            label_idx = label_idx.item()
            x1, y1, x2, y2 = map(int, box)
            class_name = CLASSES[label_idx]
            color = COLORS[label_idx].tolist()

            # Vẽ bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Vẽ text
            label_text = f"{class_name}: {score:.2f}"
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(frame, label_text, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)
        progress_bar.update(1)

    cap.release()
    out.release()
    progress_bar.close()
    print(f"✅ Done! Output saved to {args.output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Object Detection on Video")
    parser.add_argument('--model-path', type=str, default="models/c2a_best_model.pth", help="Path to the trained model")
    parser.add_argument('--video-path', type=str, default="test_video.mp4", help="Path to the input video")
    parser.add_argument('--output-path', type=str, default="output.avi", help="Path to the output video")
    parser.add_argument('--threshold', type=float, default=0.5, help="Score threshold for detection")
    args = parser.parse_args()
    
    main(args)