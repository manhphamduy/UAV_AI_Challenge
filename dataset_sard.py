import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class SARDDataset(Dataset):
    """
    Dataset cho SARD với mapping sang 11 classes của VisDrone
    Giữ nguyên head (12 classes bao gồm background)
    """
    
    # MAPPING: SARD class -> VisDrone class
    # SARD có 10 classes (0-9), VisDrone có 11 classes (1-11, 0 là background)
    SARD_TO_VISDRONE = {
        0: 1,   # ignored-regions -> pedestrian
        1: 2,   # pedestrian -> people
        2: 3,   # people -> bicycle
        3: 4,   # bicycle -> car
        4: 5,   # car -> van
        5: 6,   # cart -> tricycle
        6: 7,   # van -> awning-tricycle
        7: 8,   # truck -> bus
        8: 9,   # tricycle -> motor
        9: 10,  # awning-tricycle -> tricycle (motor vehicle)
        # Nếu có class 10 trong SARD -> map sang 11 (van)
    }
    
    def __init__(self, root_dir, transforms=None):
        """
        Args:
            root_dir: Đường dẫn đến thư mục chứa images/ và labels/
            transforms: Các phép biến đổi ảnh
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.img_dir = os.path.join(root_dir, "images")
        self.ann_dir = os.path.join(root_dir, "labels")
        
        # Lấy danh sách tất cả file ảnh
        self.images = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        print(f"✅ SARD Dataset: Found {len(self.images)} images in {root_dir}")
        print(f"ℹ️  Mapping SARD classes -> VisDrone classes (giữ nguyên 12 classes)")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        
        # Load annotations (YOLO format)
        ann_name = os.path.splitext(img_name)[0] + ".txt"
        ann_path = os.path.join(self.ann_dir, ann_name)
        
        boxes = []
        labels = []
        
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        sard_class = int(parts[0])
                        
                        # MAP sang VisDrone class
                        visdrone_class = self.SARD_TO_VISDRONE.get(sard_class, 1)  # Default: pedestrian
                        
                        # YOLO format: class x_center y_center width height (normalized)
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert to absolute coordinates
                        img_width, img_height = img.size
                        x_center_abs = x_center * img_width
                        y_center_abs = y_center * img_height
                        width_abs = width * img_width
                        height_abs = height * img_height
                        
                        # Convert to [xmin, ymin, xmax, ymax]
                        xmin = x_center_abs - width_abs / 2
                        ymin = y_center_abs - height_abs / 2
                        xmax = x_center_abs + width_abs / 2
                        ymax = y_center_abs + height_abs / 2
                        
                        # Clip to image boundaries
                        xmin = max(0, min(xmin, img_width))
                        ymin = max(0, min(ymin, img_height))
                        xmax = max(0, min(xmax, img_width))
                        ymax = max(0, min(ymax, img_height))
                        
                        # Only add valid boxes
                        if xmax > xmin and ymax > ymin:
                            boxes.append([xmin, ymin, xmax, ymax])
                            labels.append(visdrone_class)
        
        # Convert to tensors
        if len(boxes) == 0:
            # Empty image - add dummy box
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Create target dict
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }
        
        # Apply transforms
        if self.transforms:
            img = self.transforms(img)
        
        return img, target


# Test dataset
if __name__ == "__main__":
    from torchvision import transforms
    
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = SARDDataset("train", transforms=transform)
    
    print(f"\n📊 Dataset Info:")
    print(f"   Total images: {len(dataset)}")
    
    # Test một sample
    img, target = dataset[0]
    print(f"\n🖼️  Sample 0:")
    print(f"   Image shape: {img.shape}")
    print(f"   Boxes: {target['boxes'].shape}")
    print(f"   Labels: {target['labels']}")
    print(f"   Label range: {target['labels'].min()}-{target['labels'].max()}")