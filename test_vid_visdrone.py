import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset_visdrone_vid import VisDroneVideoDataset
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
    parser = argparse.ArgumentParser(description='Test VisDrone VID Model')
    parser.add_argument('--test_path', type=str, default='data/VisDrone2019-VID-test-dev',
                        help='Path to test dataset')
    parser.add_argument('--model_path', type=str, default='models/vid_best_model.pth',
                        help='Path to model')
    parser.add_argument('--output_dir', type=str, default='test_results_vid',
                        help='Output directory')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['eval', 'visualize_seq', 'visualize_random', 'stats', 'export', 'all'],
                        help='Test mode')
    parser.add_argument('--num_sequences', type=int, default=3,
                        help='Number of sequences to visualize')
    parser.add_argument('--frames_per_seq', type=int, default=10,
                        help='Number of frames per sequence')
    parser.add_argument('--num_random', type=int, default=20,
                        help='Number of random samples')
    parser.add_argument('--confidence', type=float, default=0.5,
                        help='Confidence threshold')
    
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

def visualize_video_sequences(model, test_dataset, num_sequences, frames_per_seq, output_dir):
    """Visualize video sequences"""
    print(f"\n🎬 === VISUALIZING VIDEO SEQUENCES ===")
    
    # Group images by sequence
    sequences = {}
    for img_path in test_dataset.samples:
        seq_name = os.path.basename(os.path.dirname(img_path))
        if seq_name not in sequences:
            sequences[seq_name] = []
        sequences[seq_name].append(img_path)
    
    print(f"Found {len(sequences)} sequences")
    
    seq_names = sorted(sequences.keys())[:num_sequences]
    
    for seq_name in seq_names:
        print(f"\n📹 Processing: {seq_name}")
        seq_output_dir = os.path.join(output_dir, "sequences", seq_name)
        os.makedirs(seq_output_dir, exist_ok=True)
        
        img_paths = sorted(sequences[seq_name])[:frames_per_seq]
        
        transform = transforms.Compose([transforms.ToTensor()])
        
        with torch.no_grad():
            for idx, img_path in enumerate(tqdm(img_paths, desc=f"  {seq_name}")):
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).to(device)
                
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
                        
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                        text = f"{class_name}: {score:.2f}"
                        draw.text((x1, max(0, y1 - 10)), text, fill=color)
                
                frame_name = os.path.basename(img_path)
                output_path = os.path.join(seq_output_dir, frame_name)
                img.save(output_path)
        
        print(f"  ✅ Saved {len(img_paths)} frames")

def visualize_random_samples(model, test_dataset, num_samples, output_dir):
    """Visualize random samples"""
    print(f"\n🖼️  === VISUALIZING {num_samples} RANDOM SAMPLES ===")
    
    import random
    random.seed(42)
    indices = random.sample(range(len(test_dataset)), min(num_samples, len(test_dataset)))
    
    samples_dir = os.path.join(output_dir, "random_samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    with torch.no_grad():
        for i, idx in enumerate(tqdm(indices, desc="Processing")):
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
                    class_name = CLASS_NAMES.get(label_id, 'unknown')
                    color = COLORS[label_id % len(COLORS)]
                    
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    text = f"{class_name}: {score:.2f}"
                    draw.text((x1, max(0, y1 - 10)), text, fill=color)
            
            output_path = os.path.join(samples_dir, f"sample_{i:04d}.jpg")
            img_pil.save(output_path)
    
    print(f"✅ Saved to {samples_dir}/")

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
    
    print(f"\n📈 Statistics (confidence > {confidence_threshold}):")
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
    """Export predictions to VisDrone VID format"""
    print(f"\n📝 === EXPORTING PREDICTIONS ===")
    
    predictions_dir = os.path.join(output_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Group by sequences
    sequences = {}
    for img_path in test_dataset.samples:
        seq_name = os.path.basename(os.path.dirname(img_path))
        if seq_name not in sequences:
            sequences[seq_name] = []
        sequences[seq_name].append(img_path)
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    with torch.no_grad():
        for seq_name, img_paths in tqdm(sequences.items(), desc="Exporting"):
            output_file = os.path.join(predictions_dir, f"{seq_name}.txt")
            
            with open(output_file, 'w') as f:
                for img_path in sorted(img_paths):
                    # Get frame_id
                    frame_name = os.path.basename(img_path)
                    frame_id = int(os.path.splitext(frame_name)[0])
                    
                    # Load and predict
                    img = Image.open(img_path).convert("RGB")
                    img_tensor = transform(img).to(device)
                    prediction = model([img_tensor])[0]
                    
                    # Write predictions
                    boxes = prediction['boxes'].cpu()
                    scores = prediction['scores'].cpu()
                    labels = prediction['labels'].cpu()
                    
                    target_id = 1
                    for box, score, label in zip(boxes, scores, labels):
                        if score > confidence_threshold:
                            x1, y1, x2, y2 = box.tolist()
                            w = x2 - x1
                            h = y2 - y1
                            class_id = int(label.item())
                            
                            # VisDrone VID format
                            line = f"{frame_id},{target_id},{x1:.1f},{y1:.1f},{w:.1f},{h:.1f},{score:.4f},{class_id},-1,-1\n"
                            f.write(line)
                            target_id += 1
    
    print(f"✅ Saved to {predictions_dir}/")
    print(f"   Total sequences: {len(sequences)}")

# ======================================================================
# MAIN
# ======================================================================
def main():
    args = parse_args()
    
    print("="*60)
    print("🚀 VISDRONE VIDEO MODEL TESTING (KAGGLE)")
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
    test_dataset = VisDroneVideoDataset(args.test_path, transforms=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                             collate_fn=lambda x: tuple(zip(*x)))
    print(f"✅ Found {len(test_dataset)} images")
    
    # Run based on mode
    if args.mode == 'eval':
        run_evaluation(model, test_loader, args.output_dir, args.test_path, args.model_path)
    
    elif args.mode == 'visualize_seq':
        visualize_video_sequences(model, test_dataset, args.num_sequences, 
                                 args.frames_per_seq, args.output_dir)
    
    elif args.mode == 'visualize_random':
        visualize_random_samples(model, test_dataset, args.num_random, args.output_dir)
    
    elif args.mode == 'stats':
        compute_statistics(model, test_loader, args.output_dir)
    
    elif args.mode == 'export':
        export_predictions(model, test_dataset, args.output_dir)
    
    elif args.mode == 'all':
        print("\n🔄 Running all modes...\n")
        try:
            run_evaluation(model, test_loader, args.output_dir, args.test_path, args.model_path)
        except Exception as e:
            print(f"⚠️  Evaluation failed: {e}")
        
        visualize_video_sequences(model, test_dataset, args.num_sequences, 
                                 args.frames_per_seq, args.output_dir)
        visualize_random_samples(model, test_dataset, args.num_random, args.output_dir)
        compute_statistics(model, test_loader, args.output_dir)
        export_predictions(model, test_dataset, args.output_dir)
    
    print("\n" + "="*60)
    print("✅ DONE!")
    print(f"📂 Results: {args.output_dir}/")
    print("="*60)

if __name__ == "__main__":
    main()