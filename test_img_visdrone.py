import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset_visdrone_det import VisDroneDetDataset
from evaluate import evaluate_model
import json
import argparse

# ======================================================================
# CONFIG
# ======================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 12
confidence_threshold = 0.5

# VisDrone class names
CLASS_NAMES = {
    0: 'background', 1: 'pedestrian', 2: 'people', 3: 'bicycle',
    4: 'car', 5: 'van', 6: 'truck', 7: 'tricycle',
    8: 'awning-tricycle', 9: 'bus', 10: 'motor', 11: 'others'
}

COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
]

# ======================================================================
# PARSE ARGUMENTS
# ======================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Test VisDrone IMG Model')
    parser.add_argument('--test_path', type=str, default='data/VisDrone2019-DET-test-dev',
                        help='Path to test dataset')
    parser.add_argument('--model_path', type=str, default='models/img_best_model.pth',
                        help='Path to model')
    parser.add_argument('--output_dir', type=str, default='test_results_img',
                        help='Output directory')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['eval', 'visualize', 'predict', 'stats', 'export', 'compare', 'all'],
                        help='Test mode')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples to visualize')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Single image path for prediction')
    
    return parser.parse_args()

# ======================================================================
# FUNCTIONS
# ======================================================================
def load_model(model_path):
    """Load model"""
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
    return model

def run_evaluation(model, test_loader, output_dir, test_path, model_path):
    """Đánh giá mAP"""
    print("\n📊 === EVALUATION ===")
    mAP = evaluate_model(model, test_loader, device)
    print(f"🎯 Test mAP: {mAP:.4f}")
    
    result_path = os.path.join(output_dir, "evaluation_result.txt")
    with open(result_path, 'w') as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Test set: {test_path}\n")
        f.write(f"mAP: {mAP:.4f}\n")
    
    print(f"💾 Saved to {result_path}")
    return mAP

def visualize_predictions(model, test_dataset, num_samples, output_dir):
    """Visualize predictions"""
    print(f"\n🖼️  === VISUALIZING {num_samples} SAMPLES ===")
    
    samples_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(samples_dir, exist_ok=True)
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    with torch.no_grad():
        for idx in tqdm(range(min(num_samples, len(test_dataset)))):
            img, target = test_dataset[idx]
            img_tensor = img.to(device)
            
            prediction = model([img_tensor])[0]
            
            img_pil = transforms.ToPILImage()(img.cpu())
            draw = ImageDraw.Draw(img_pil)
            
            boxes = prediction['boxes'].cpu()
            scores = prediction['scores'].cpu()
            labels = prediction['labels'].cpu()
            
            for box, score, label in zip(boxes, scores, labels):
                if score > confidence_threshold:
                    x1, y1, x2, y2 = box.tolist()
                    label_id = int(label.item())
                    class_name = CLASS_NAMES.get(label_id, 'unknown')
                    color = COLORS[label_id % len(COLORS)]
                    
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    text = f"{class_name}: {score:.2f}"
                    draw.text((x1, max(0, y1 - 12)), text, fill=color)
            
            img_name = test_dataset.img_files[idx]
            output_name = f"{idx:04d}_{os.path.basename(img_name)}"
            output_path = os.path.join(samples_dir, output_name)
            img_pil.save(output_path)
    
    print(f"✅ Saved to {samples_dir}/")

def predict_single_image(model, image_path, output_dir):
    """Predict single image"""
    print(f"\n🖼️  === PREDICTING SINGLE IMAGE ===")
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    img = Image.open(image_path).convert("RGB")
    img_tensor = transforms.ToTensor()(img).to(device)
    
    with torch.no_grad():
        prediction = model([img_tensor])[0]
    
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
    
    output_path = os.path.join(output_dir, "single_prediction.jpg")
    img.save(output_path)
    print(f"💾 Saved to {output_path}")

def compute_statistics(model, test_loader, output_dir):
    """Compute statistics"""
    print(f"\n📊 === COMPUTING STATISTICS ===")
    
    class_counts = {i: 0 for i in range(num_classes)}
    total_detections = 0
    total_images = 0
    images_with_detections = 0
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Analyzing"):
            total_images += 1
            images_gpu = [img.to(device) for img in images]
            predictions = model(images_gpu)
            
            pred = predictions[0]
            scores = pred['scores'].cpu()
            labels = pred['labels'].cpu()
            
            has_detection = False
            for score, label in zip(scores, labels):
                if score > confidence_threshold:
                    has_detection = True
                    label_id = int(label.item())
                    class_counts[label_id] += 1
                    total_detections += 1
            
            if has_detection:
                images_with_detections += 1
    
    print(f"\n📈 Statistics:")
    print(f"   Total images: {total_images}")
    print(f"   Images with detections: {images_with_detections} ({images_with_detections/total_images*100:.1f}%)")
    print(f"   Total detections: {total_detections}")
    print(f"   Avg detections/image: {total_detections/total_images:.2f}")
    
    print(f"\n   Per class:")
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        if count > 0:
            class_name = CLASS_NAMES.get(class_id, 'unknown')
            pct = (count / total_detections * 100) if total_detections > 0 else 0
            print(f"      {class_name:20s}: {count:6d} ({pct:5.2f}%)")
    
    stats = {
        'total_images': total_images,
        'images_with_detections': images_with_detections,
        'total_detections': total_detections,
        'avg_detections_per_image': total_detections / total_images,
        'class_counts': {CLASS_NAMES[k]: v for k, v in class_counts.items() if v > 0}
    }
    
    stats_path = os.path.join(output_dir, "statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"💾 Saved to {stats_path}")

def export_predictions(model, test_dataset, output_dir):
    """Export predictions"""
    print(f"\n📝 === EXPORTING PREDICTIONS ===")
    
    predictions_dir = os.path.join(output_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset))):
            img, target = test_dataset[idx]
            img_tensor = img.to(device)
            
            prediction = model([img_tensor])[0]
            
            img_name = test_dataset.img_files[idx]
            txt_name = os.path.splitext(os.path.basename(img_name))[0] + ".txt"
            txt_path = os.path.join(predictions_dir, txt_name)
            
            boxes = prediction['boxes'].cpu()
            scores = prediction['scores'].cpu()
            labels = prediction['labels'].cpu()
            
            with open(txt_path, 'w') as f:
                for box, score, label in zip(boxes, scores, labels):
                    if score > confidence_threshold:
                        x1, y1, x2, y2 = box.tolist()
                        w = x2 - x1
                        h = y2 - y1
                        class_id = int(label.item())
                        line = f"{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},{score:.4f},{class_id},-1,-1\n"
                        f.write(line)
    
    print(f"✅ Saved to {predictions_dir}/")

def compare_with_ground_truth(model, test_dataset, num_samples, output_dir):
    """Compare predictions with ground truth"""
    print(f"\n🔍 === COMPARING WITH GT ===")
    
    compare_dir = os.path.join(output_dir, "comparisons")
    os.makedirs(compare_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx in tqdm(range(min(num_samples, len(test_dataset)))):
            img, target = test_dataset[idx]
            img_tensor = img.to(device)
            
            prediction = model([img_tensor])[0]
            
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            
            # Ground Truth
            img_gt = transforms.ToPILImage()(img.cpu())
            draw_gt = ImageDraw.Draw(img_gt)
            
            if len(target['boxes']) > 0:
                for box, label in zip(target['boxes'], target['labels']):
                    x1, y1, x2, y2 = box.tolist()
                    label_id = int(label.item())
                    class_name = CLASS_NAMES.get(label_id, 'unknown')
                    color = COLORS[label_id % len(COLORS)]
                    draw_gt.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    draw_gt.text((x1, y1 - 10), class_name, fill=color)
            
            axes[0].imshow(img_gt)
            axes[0].set_title(f"GT ({len(target['boxes'])} objects)", fontsize=14)
            axes[0].axis('off')
            
            # Predictions
            img_pred = transforms.ToPILImage()(img.cpu())
            draw_pred = ImageDraw.Draw(img_pred)
            
            boxes = prediction['boxes'].cpu()
            scores = prediction['scores'].cpu()
            labels = prediction['labels'].cpu()
            
            num_preds = 0
            for box, score, label in zip(boxes, scores, labels):
                if score > confidence_threshold:
                    num_preds += 1
                    x1, y1, x2, y2 = box.tolist()
                    label_id = int(label.item())
                    class_name = CLASS_NAMES.get(label_id, 'unknown')
                    color = COLORS[label_id % len(COLORS)]
                    draw_pred.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    text = f"{class_name}: {score:.2f}"
                    draw_pred.text((x1, y1 - 10), text, fill=color)
            
            axes[1].imshow(img_pred)
            axes[1].set_title(f"Predictions ({num_preds} objects)", fontsize=14)
            axes[1].axis('off')
            
            plt.tight_layout()
            
            img_name = test_dataset.img_files[idx]
            output_name = f"{idx:04d}_compare.jpg"
            output_path = os.path.join(compare_dir, output_name)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"✅ Saved to {compare_dir}/")

# ======================================================================
# MAIN
# ======================================================================
def main():
    args = parse_args()
    
    print("="*60)
    print("🚀 VISDRONE IMAGE MODEL TESTING (KAGGLE)")
    print("="*60)
    print(f"\nModel: {args.model_path}")
    print(f"Test set: {args.test_path}")
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Confidence: {args.confidence}")
    
    global confidence_threshold
    confidence_threshold = args.confidence
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.model_path)
    
    # Load dataset
    print(f"\n📂 Loading dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = VisDroneDetDataset(args.test_path, transforms=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                             collate_fn=lambda x: tuple(zip(*x)))
    print(f"✅ Found {len(test_dataset)} images")
    
    # Run based on mode
    if args.mode == 'eval':
        run_evaluation(model, test_loader, args.output_dir, args.test_path, args.model_path)
    
    elif args.mode == 'visualize':
        visualize_predictions(model, test_dataset, args.num_samples, args.output_dir)
    
    elif args.mode == 'predict':
        if args.image_path:
            predict_single_image(model, args.image_path, args.output_dir)
        else:
            print("❌ Please provide --image_path")
    
    elif args.mode == 'stats':
        compute_statistics(model, test_loader, args.output_dir)
    
    elif args.mode == 'export':
        export_predictions(model, test_dataset, args.output_dir)
    
    elif args.mode == 'compare':
        compare_with_ground_truth(model, test_dataset, args.num_samples, args.output_dir)
    
    elif args.mode == 'all':
        print("\n🔄 Running all modes...\n")
        try:
            run_evaluation(model, test_loader, args.output_dir, args.test_path, args.model_path)
        except Exception as e:
            print(f"⚠️  Evaluation failed: {e}")
        
        visualize_predictions(model, test_dataset, args.num_samples, args.output_dir)
        compute_statistics(model, test_loader, args.output_dir)
        export_predictions(model, test_dataset, args.output_dir)
        compare_with_ground_truth(model, test_dataset, min(10, args.num_samples), args.output_dir)
    
    print("\n" + "="*60)
    print("✅ DONE!")
    print(f"📂 Results: {args.output_dir}/")
    print("="*60)

if __name__ == "__main__":
    main()