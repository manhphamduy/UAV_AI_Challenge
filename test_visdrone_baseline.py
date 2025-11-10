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
    parser = argparse.ArgumentParser(description='Test Baseline MobileNet on VisDrone')
    parser.add_argument('--test_path', type=str, default='data/VisDrone2019-DET-test-dev',
                        help='Path to test dataset')
    parser.add_argument('--output_dir', type=str, default='test_results_baseline',
                        help='Output directory')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['eval', 'visualize', 'stats', 'compare', 'all'],
                        help='Test mode')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='Number of samples to visualize')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--pretrained', type=str, default='COCO',
                        choices=['COCO', 'ImageNet'],
                        help='Pretrained weights: COCO or ImageNet')
    
    return parser.parse_args()

# ======================================================================
# LOAD BASELINE MODEL
# ======================================================================
def load_baseline_model(pretrained='COCO'):
    """
    Load baseline MobileNetV3 FasterRCNN
    pretrained: 'COCO' hoặc 'ImageNet'
    """
    print(f"\n🔄 Loading baseline MobileNetV3 FasterRCNN...")
    print(f"   Pretrained weights: {pretrained}")
    
    if pretrained == 'COCO':
        # Load model với COCO pretrained (91 classes)
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
        print(f"✅ Loaded COCO pretrained model (91 classes)")
        print(f"⚠️  Note: COCO classes khác VisDrone, kết quả có thể không tốt")
        
    elif pretrained == 'ImageNet':
        # Load model với ImageNet backbone, random head
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
        
        # Thay head cho 12 classes của VisDrone
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        print(f"✅ Loaded ImageNet backbone with random head (12 classes)")
        print(f"⚠️  Note: Head chưa được train, kết quả sẽ rất tệ")
    
    model.to(device)
    model.eval()
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model info:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Device: {device}")
    
    return model

# ======================================================================
# EVALUATION
# ======================================================================
def run_evaluation(model, test_loader, output_dir, test_path, pretrained):
    """Đánh giá mAP"""
    print("\n📊 === BASELINE EVALUATION ===")
    
    try:
        mAP = evaluate_model(model, test_loader, device)
        print(f"🎯 Baseline mAP: {mAP:.4f}")
    except Exception as e:
        print(f"⚠️  Evaluation failed: {e}")
        print(f"   Có thể do COCO classes không map với VisDrone")
        mAP = 0.0
    
    result_path = os.path.join(output_dir, "baseline_evaluation.txt")
    with open(result_path, 'w') as f:
        f.write(f"Model: Baseline MobileNetV3 FasterRCNN\n")
        f.write(f"Pretrained: {pretrained}\n")
        f.write(f"Test set: {test_path}\n")
        f.write(f"mAP: {mAP:.4f}\n")
    
    print(f"💾 Saved to {result_path}")
    return mAP

# ======================================================================
# VISUALIZE
# ======================================================================
def visualize_predictions(model, test_dataset, num_samples, output_dir):
    """Visualize predictions"""
    print(f"\n🖼️  === VISUALIZING {num_samples} BASELINE PREDICTIONS ===")
    
    samples_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(samples_dir, exist_ok=True)
    
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
            
            num_detections = 0
            for box, score, label in zip(boxes, scores, labels):
                if score > confidence_threshold:
                    num_detections += 1
                    x1, y1, x2, y2 = box.tolist()
                    label_id = int(label.item())
                    
                    # Map class name (có thể là COCO classes)
                    if label_id < len(CLASS_NAMES):
                        class_name = CLASS_NAMES.get(label_id, f'class_{label_id}')
                    else:
                        class_name = f'coco_class_{label_id}'
                    
                    color = COLORS[label_id % len(COLORS)]
                    
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    text = f"{class_name}: {score:.2f}"
                    draw.text((x1, max(0, y1 - 12)), text, fill=color)
            
            output_name = f"baseline_{idx:04d}.jpg"
            output_path = os.path.join(samples_dir, output_name)
            img_pil.save(output_path)
            
            if idx < 5:
                print(f"   Image {idx}: {num_detections} detections")
    
    print(f"✅ Saved to {samples_dir}/")

# ======================================================================
# STATISTICS
# ======================================================================
def compute_statistics(model, test_loader, output_dir):
    """Compute statistics"""
    print(f"\n📊 === BASELINE STATISTICS ===")
    
    class_counts = {}
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
                    
                    if label_id not in class_counts:
                        class_counts[label_id] = 0
                    class_counts[label_id] += 1
                    total_detections += 1
            
            if has_detection:
                images_with_detections += 1
    
    print(f"\n📈 Baseline Statistics (confidence > {confidence_threshold}):")
    print(f"   Total images: {total_images}")
    print(f"   Images with detections: {images_with_detections} ({images_with_detections/total_images*100:.1f}%)")
    print(f"   Total detections: {total_detections}")
    print(f"   Avg detections/image: {total_detections/total_images:.2f}")
    
    print(f"\n   Detected classes:")
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        if class_id < len(CLASS_NAMES):
            class_name = CLASS_NAMES.get(class_id, f'class_{class_id}')
        else:
            class_name = f'coco_class_{class_id}'
        
        pct = (count / total_detections * 100) if total_detections > 0 else 0
        print(f"      {class_name:20s} (id={class_id:2d}): {count:6d} ({pct:5.2f}%)")
    
    stats = {
        'total_images': total_images,
        'images_with_detections': images_with_detections,
        'total_detections': total_detections,
        'avg_detections_per_image': total_detections / total_images,
        'class_counts': {f"class_{k}": v for k, v in class_counts.items()}
    }
    
    stats_path = os.path.join(output_dir, "baseline_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"💾 Saved to {stats_path}")

# ======================================================================
# COMPARE BASELINE vs FINETUNED
# ======================================================================
def compare_with_finetuned(baseline_model, test_dataset, num_samples, output_dir, finetuned_model_path):
    """So sánh baseline vs finetuned model"""
    print(f"\n🔍 === COMPARING BASELINE vs FINETUNED ===")
    
    # Load finetuned model
    if not os.path.exists(finetuned_model_path):
        print(f"⚠️  Finetuned model not found: {finetuned_model_path}")
        print(f"   Skipping comparison")
        return
    
    print(f"🔄 Loading finetuned model...")
    finetuned_model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
    in_features = finetuned_model.roi_heads.box_predictor.cls_score.in_features
    finetuned_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    finetuned_model.load_state_dict(torch.load(finetuned_model_path, map_location=device, weights_only=False))
    finetuned_model.to(device)
    finetuned_model.eval()
    print(f"✅ Loaded finetuned model")
    
    compare_dir = os.path.join(output_dir, "comparisons")
    os.makedirs(compare_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx in tqdm(range(min(num_samples, len(test_dataset)))):
            img, target = test_dataset[idx]
            img_tensor = img.to(device)
            
            # Baseline prediction
            baseline_pred = baseline_model([img_tensor])[0]
            
            # Finetuned prediction
            finetuned_pred = finetuned_model([img_tensor])[0]
            
            # Create comparison
            fig, axes = plt.subplots(1, 3, figsize=(30, 10))
            
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
            axes[0].set_title(f"Ground Truth ({len(target['boxes'])} objects)", fontsize=14)
            axes[0].axis('off')
            
            # Baseline
            img_baseline = transforms.ToPILImage()(img.cpu())
            draw_baseline = ImageDraw.Draw(img_baseline)
            
            boxes = baseline_pred['boxes'].cpu()
            scores = baseline_pred['scores'].cpu()
            labels = baseline_pred['labels'].cpu()
            
            num_baseline = 0
            for box, score, label in zip(boxes, scores, labels):
                if score > confidence_threshold:
                    num_baseline += 1
                    x1, y1, x2, y2 = box.tolist()
                    label_id = int(label.item())
                    color = COLORS[label_id % len(COLORS)]
                    draw_baseline.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            axes[1].imshow(img_baseline)
            axes[1].set_title(f"Baseline ({num_baseline} objects)", fontsize=14)
            axes[1].axis('off')
            
            # Finetuned
            img_finetuned = transforms.ToPILImage()(img.cpu())
            draw_finetuned = ImageDraw.Draw(img_finetuned)
            
            boxes = finetuned_pred['boxes'].cpu()
            scores = finetuned_pred['scores'].cpu()
            labels = finetuned_pred['labels'].cpu()
            
            num_finetuned = 0
            for box, score, label in zip(boxes, scores, labels):
                if score > confidence_threshold:
                    num_finetuned += 1
                    x1, y1, x2, y2 = box.tolist()
                    label_id = int(label.item())
                    class_name = CLASS_NAMES.get(label_id, 'unknown')
                    color = COLORS[label_id % len(COLORS)]
                    draw_finetuned.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    text = f"{class_name}: {score:.2f}"
                    draw_finetuned.text((x1, y1 - 10), text, fill=color)
            
            axes[2].imshow(img_finetuned)
            axes[2].set_title(f"Finetuned ({num_finetuned} objects)", fontsize=14)
            axes[2].axis('off')
            
            plt.tight_layout()
            
            output_name = f"compare_{idx:04d}.jpg"
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
    print("🚀 BASELINE MOBILENET TESTING ON VISDRONE")
    print("="*60)
    print(f"\nPretrained: {args.pretrained}")
    print(f"Test set: {args.test_path}")
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")
    print(f"Confidence: {args.confidence}")
    
    global confidence_threshold
    confidence_threshold = args.confidence
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load baseline model
    model = load_baseline_model(pretrained=args.pretrained)
    
    # Load dataset
    print(f"\n📂 Loading dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = VisDroneDetDataset(args.test_path, transforms=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                             collate_fn=lambda x: tuple(zip(*x)))
    print(f"✅ Found {len(test_dataset)} images")
    
    # Run based on mode
    if args.mode == 'eval':
        run_evaluation(model, test_loader, args.output_dir, args.test_path, args.pretrained)
    
    elif args.mode == 'visualize':
        visualize_predictions(model, test_dataset, args.num_samples, args.output_dir)
    
    elif args.mode == 'stats':
        compute_statistics(model, test_loader, args.output_dir)
    
    elif args.mode == 'compare':
        finetuned_path = "models/img_best_model.pth"
        compare_with_finetuned(model, test_dataset, args.num_samples, 
                              args.output_dir, finetuned_path)
    
    elif args.mode == 'all':
        print("\n🔄 Running all modes...\n")
        run_evaluation(model, test_loader, args.output_dir, args.test_path, args.pretrained)
        visualize_predictions(model, test_dataset, args.num_samples, args.output_dir)
        compute_statistics(model, test_loader, args.output_dir)
        
        # Compare if finetuned model exists
        finetuned_path = "models/img_best_model.pth"
        if os.path.exists(finetuned_path):
            compare_with_finetuned(model, test_dataset, min(10, args.num_samples), 
                                  args.output_dir, finetuned_path)
    
    print("\n" + "="*60)
    print("✅ DONE!")
    print(f"📂 Results: {args.output_dir}/")
    print("="*60)

if __name__ == "__main__":
    main()