import cv2
import torch
import torchvision
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ======================================================================
# PHẦN CẤU HÌNH - CHỈ CẦN CHỈNH SỬA Ở ĐÂY
# ======================================================================
class Config:
    MODEL_PATH = "models/best_model_det_final.pth"
    IMAGE_PATH = "sample_frame.jpg"
    OUTPUT_PATH = "gradcam_overlay.jpg"
    TARGET_CLASS = None

CLASSES = [
    '__background__', 'pedestrian', 'person', 'car', 'van', 'bus', 
    'truck', 'motor', 'bicycle', 'awning-tricycle', 'tricycle', 'other'
]

def generate_gradcam(model, img_tensor, target_class_idx=None):
    gradients = []
    activations = []
    
    def save_gradients(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    def save_activations(module, input, output):
        activations.append(output.detach())

    # ===== SỬA LỖI: Dùng list(children()) để lấy layer cuối =====
    body_layers = list(model.backbone.body.children())
    if len(body_layers) == 0:
        raise ValueError("Backbone body rỗng!")
    
    final_layer = body_layers[-1]
    # Giả sử bạn muốn hook vào block đầu tiên của layer cuối (thường là Conv2d)
    target_layer = final_layer.block[0] if hasattr(final_layer, 'block') else final_layer
    # =============================================================
    
    forward_handle = target_layer.register_forward_hook(save_activations)
    backward_handle = target_layer.register_full_backward_hook(save_gradients)
    
    try:
        model.zero_grad()
        output = model([img_tensor])[0]

        if len(output["scores"]) == 0:
            print("Cảnh báo: Model không phát hiện được vật thể nào trong ảnh.")
            return None, None

        if target_class_idx is not None:
            indices = [i for i, label in enumerate(output["labels"]) if label == target_class_idx]
            if not indices:
                print(f"Cảnh báo: Model không phát hiện được vật thể nào thuộc lớp {CLASSES[target_class_idx]}.")
                return None, None
            scores_for_class = output["scores"][indices]
            best_idx_for_class = indices[scores_for_class.argmax()]
            target_score = output["scores"][best_idx_for_class]
            target_box = output["boxes"][best_idx_for_class]
        else:
            best_idx = output["scores"].argmax()
            target_score = output["scores"][best_idx]
            target_box = output["boxes"][best_idx]

        target_score.backward()
        
        grad = gradients[0].cpu().numpy()[0]
        act = activations[0].cpu().numpy()[0]
        
        weights = np.mean(grad, axis=(1, 2))
        cam = np.zeros(act.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * act[i, :, :]

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam, target_box.detach().cpu().numpy()

    finally:
        forward_handle.remove()
        backward_handle.remove()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")

    print("Đang tải model...")
    num_classes = len(CLASSES)
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.to(device).eval()
    print("✅ Model đã được tải thành công.")

    img = cv2.imread(Config.IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Không thể đọc ảnh tại: {Config.IMAGE_PATH}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torchvision.transforms.functional.to_tensor(img_rgb).to(device)

    cam, target_box = generate_gradcam(model, img_tensor, Config.TARGET_CLASS)

    if cam is None:
        print("Không thể tạo Grad-CAM.")
        return

    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    x1, y1, x2, y2 = map(int, target_box)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.imwrite(Config.OUTPUT_PATH, overlay)
    print(f"✅ Ảnh Grad-CAM đã được lưu tại: {Config.OUTPUT_PATH}")

if __name__ == '__main__':
    main()