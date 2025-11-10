# dataset_visdrone_det.py (PHIÊN BẢN SỬA LỖI TRIỆT ĐỂ)
import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class VisDroneDetDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.img_dir = os.path.join(root_dir, "images")
        self.ann_dir = os.path.join(root_dir, "annotations")
        
        # Lấy danh sách file ảnh
        self.img_files = sorted([
            os.path.join(self.img_dir, f) 
            for f in os.listdir(self.img_dir) 
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ])
        
        print(f"✅ Found {len(self.img_files)} images in {root_dir}")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert("RGB")
        
        # Load annotation
        img_name = os.path.basename(img_path)
        ann_name = os.path.splitext(img_name)[0] + ".txt"
        ann_path = os.path.join(self.ann_dir, ann_name)
        
        boxes = []
        labels = []
        
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 8:
                        x, y, w, h = map(float, parts[0:4])
                        cls = int(parts[7])
                        
                        if cls == 0 or w <= 0 or h <= 0:
                            continue
                        
                        boxes.append([x, y, x + w, y + h])
                        labels.append(cls)
        
        # Convert to tensors
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }
        
        if self.transforms:
            img = self.transforms(img)
        
        return img, target