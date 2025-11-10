import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset_c2a import C2ADataset
from evaluate import evaluate_model

# ======================================================================
# CONFIG
# ======================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 12
test_path = "data/c2a/test"
model_path = "models/c2a_best_model.pth"  # Hoặc sard_best_model.pth
output_dir = "data/c2a/test_results"
confidence_threshold = 0.5  # Chỉ hiển thị detection có confidence > 0.5

os.makedirs(output_dir, exist_ok=True)

# VisDrone class names
CLASS_NAMES = {
    0: 'background',
    1: 'pedestrian',
    2: 'people',
    3: 'bicycle',
    4: 'car',
    5: 'van',
    6: 'truck',
    7: 'tricycle',
    8: 'awning-tricycle',
    9: 'bus',
    10: 'motor',
    11: 'others'
}

# Colors for visualization
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
]

# ======================================================================
# LOAD MODEL
# ======================================================================
print(f"🔄 Loading model from {model_path}")
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    print(f"✅ Model loaded successfully")
else:
    raise FileNotFoundError(f"❌ Model not found: {model_path}")

model.to(device)
model.eval()

# ======================================================================
# LOAD TEST DATASET
# ======================================================================
print(f"\n📂 Loading test dataset from {test_path}")
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = C2ADataset(test_path, transforms=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                         collate_fn=lambda x: tuple(zip(*x)))

# ======================================================================
# OPTION 1: EVALUATE (nếu có ground truth labels)
# ======================================================================
def run_evaluation():
    print("\n📊 === ĐÁNH GIÁ MODEL TRÊN TEST SET ===")
    mAP = evaluate_model(model, test_loader, device)
    print(f"\n🎯 Test mAP: {mAP:.4f}")
    return mAP

# ======================================================================
# OPTION 2: VISUALIZE PREDICTIONS
# ======================================================================
def visualize_predictions(num_samples=10, save_images=True):
    print(f"\n🖼️  === VISUALIZE {num_samples} SAMPLES ===")
    
    model.eval()
    with torch.no_grad():
        for idx, (images, targets) in enumerate(tqdm(test_loader, desc="Predicting")):
            if idx >= num_samples:
                break
            
            # Get prediction
            images_gpu = [img.to(device) for img in images]
            predictions = model(images_gpu)
            
            # Convert image to PIL
            img_tensor = images[0].cpu()
            img_pil = transforms.ToPILImage()(img_tensor)
            draw = ImageDraw.Draw(img_pil)
            
            # Draw predictions
            pred = predictions[0]
            boxes = pred['boxes'].cpu()
            scores = pred['scores'].cpu()
            labels = pred['labels'].cpu()
            
            num_detections = 0
            for box, score, label in zip(boxes, scores, labels):
                if score > confidence_threshold:
                    num_detections += 1
                    x1, y1, x2, y2 = box.tolist()
                    label_id = int(label.item())
                    class_name = CLASS_NAMES.get(label_id, 'unknown')
                    color = COLORS[label_id % len(COLORS)]
                    
                    # Draw box
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    
                    # Draw label
                    text = f"{class_name}: {score:.2f}"
                    draw.text((x1, y1 - 10), text, fill=color)
            
            print(f"   Image {idx}: {num_detections} detections")
            
            # Save or display
            if save_images:
                output_path = os.path.join(output_dir, f"result_{idx:04d}.jpg")
                img_pil.save(output_path)
            else:
                plt.figure(figsize=(12, 8))
                plt.imshow(img_pil)
                plt.axis('off')
                plt.title(f"Image {idx}: {num_detections} detections")
                plt.show()
    
    if save_images:
        print(f"\n✅ Results saved to {output_dir}/")

# ======================================================================
# OPTION 3: INFERENCE ON SINGLE IMAGE
# ======================================================================
def predict_single_image(image_path, save_path=None, show=True):
    """
    Dự đoán trên 1 ảnh duy nhất
    """
    print(f"\n🖼️  Predicting on: {image_path}")
    
    # Load image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transforms.ToTensor()(img).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        prediction = model([img_tensor])[0]
    
    # Draw results
    draw = ImageDraw.Draw(img)
    boxes = prediction['boxes'].cpu()
    scores = prediction['scores'].cpu()
    labels = prediction['labels'].cpu()
    
    num_detections = 0
    for box, score, label in zip(boxes, scores, labels):
        if score > confidence_threshold:
            num_detections += 1
            x1, y1, x2, y2 = box.tolist()
            label_id = int(label.item())
            class_name = CLASS_NAMES.get(label_id, 'unknown')
            color = COLORS[label_id % len(COLORS)]
            
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            text = f"{class_name}: {score:.2f}"
            draw.text((x1, y1 - 10), text, fill=color)
    
    print(f"✅ Found {num_detections} objects")
    
    # Save or show
    if save_path:
        img.save(save_path)
        print(f"💾 Saved to: {save_path}")
    
    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Detections: {num_detections}")
        plt.show()
    
    return prediction

# ======================================================================
# OPTION 4: EXPORT PREDICTIONS TO FILE
# ======================================================================
def export_predictions_to_txt():
    """
    Xuất predictions ra file txt (YOLO format hoặc COCO format)
    """
    print(f"\n📝 === EXPORTING PREDICTIONS ===")
    
    predictions_dir = os.path.join(output_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        for idx, (images, targets) in enumerate(tqdm(test_loader, desc="Exporting")):
            images_gpu = [img.to(device) for img in images]
            predictions = model(images_gpu)
            
            # Get image dimensions
            img_tensor = images[0]
            _, img_h, img_w = img_tensor.shape
            
            # Get predictions
            pred = predictions[0]
            boxes = pred['boxes'].cpu()
            scores = pred['scores'].cpu()
            labels = pred['labels'].cpu()
            
            # Save to txt file
            img_name = test_dataset.images[idx]
            txt_name = os.path.splitext(img_name)[0] + ".txt"
            txt_path = os.path.join(predictions_dir, txt_name)
            
            with open(txt_path, 'w') as f:
                for box, score, label in zip(boxes, scores, labels):
                    if score > confidence_threshold:
                        x1, y1, x2, y2 = box.tolist()
                        
                        # Convert to YOLO format (normalized)
                        x_center = ((x1 + x2) / 2) / img_w
                        y_center = ((y1 + y2) / 2) / img_h
                        width = (x2 - x1) / img_w
                        height = (y2 - y1) / img_h
                        
                        class_id = int(label.item()) - 1  # Convert back to 0-based
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.4f}\n")
    
    print(f"✅ Predictions exported to {predictions_dir}/")

# ======================================================================
# OPTION 5: STATISTICS
# ======================================================================
def print_statistics():
    """
    Thống kê số lượng detections theo class
    """
    print(f"\n📊 === STATISTICS ===")
    
    class_counts = {i: 0 for i in range(num_classes)}
    total_detections = 0
    
    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Analyzing"):
            images_gpu = [img.to(device) for img in images]
            predictions = model(images_gpu)
            
            pred = predictions[0]
            scores = pred['scores'].cpu()
            labels = pred['labels'].cpu()
            
            for score, label in zip(scores, labels):
                if score > confidence_threshold:
                    label_id = int(label.item())
                    class_counts[label_id] += 1
                    total_detections += 1
    
    print(f"\n📈 Detection Statistics (confidence > {confidence_threshold}):")
    print(f"   Total detections: {total_detections}")
    print(f"\n   Per class:")
    for class_id, count in sorted(class_counts.items()):
        if count > 0:
            class_name = CLASS_NAMES.get(class_id, 'unknown')
            percentage = (count / total_detections * 100) if total_detections > 0 else 0
            print(f"      {class_name:20s}: {count:5d} ({percentage:5.2f}%)")

# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    print("="*60)
    print("🚀 C2A MODEL TESTING")
    print("="*60)
    
    # Chọn chế độ test
    print("\nChọn chế độ test:")
    print("1. Đánh giá mAP (cần có ground truth labels)")
    print("2. Visualize predictions (lưu ảnh)")
    print("3. Predict 1 ảnh cụ thể")
    print("4. Export predictions to txt")
    print("5. Thống kê detections")
    print("6. Chạy tất cả")
    
    import sys

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("Nhập lựa chọn (1-6): ")

    
    if choice == "1":
        run_evaluation()
    
    elif choice == "2":
        num_samples = int(input("Số lượng ảnh muốn visualize (default=10): ") or "10")
        visualize_predictions(num_samples=num_samples, save_images=True)
    
    elif choice == "3":
        img_path = input("Đường dẫn ảnh: ").strip()
        save_path = os.path.join(output_dir, "single_prediction.jpg")
        predict_single_image(img_path, save_path=save_path, show=True)
    
    elif choice == "4":
        export_predictions_to_txt()
    
    elif choice == "5":
        print_statistics()
    
    elif choice == "6":
        print("\n🔄 Chạy tất cả các chế độ...\n")
        try:
            run_evaluation()
        except:
            print("⚠️  Không thể đánh giá mAP (có thể thiếu ground truth)")
        
        visualize_predictions(num_samples=10, save_images=True)
        export_predictions_to_txt()
        print_statistics()
    
    else:
        print("❌ Lựa chọn không hợp lệ")
    
    print("\n" + "="*60)
    print("✅ HOÀN THÀNH!")
    print("="*60)