import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class C2ADataset(Dataset):
    """
    Dataset cho C2A với mapping sang 11 classes của VisDrone
    Giữ nguyên head (12 classes bao gồm background)
    """
    
    # MAPPING: C2A class -> VisDrone class
    # Điều chỉnh mapping này theo dataset C2A thực tế của bạn
    C2A_TO_VISDRONE = {
        0: 1,   # person -> pedestrian
        1: 2,   # rider -> people
        2: 4,   # car -> car
        3: 8,   # bus -> bus
        4: 6,   # truck -> truck
        5: 9,   # motorcycle -> motor
        6: 3,   # bicycle -> bicycle
        7: 5,   # van -> van
        8: 7,   # tricycle -> tricycle
        9: 10,  # other-vehicle -> motor
        # Thêm mapping nếu C2A có nhiều classes hơn
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
        if not os.path.exists(self.img_dir):
            raise ValueError(f"❌ Không tìm thấy thư mục: {self.img_dir}")
        
        self.images = sorted([f for f in os.listdir(self.img_dir) 
                             if f.endswith(('.jpg', '.png', '.jpeg', '.JPG', '.PNG'))])
        
        print(f"✅ C2A Dataset: Found {len(self.images)} images in {root_dir}")
        print(f"ℹ️  Mapping C2A classes -> VisDrone classes (giữ nguyên 12 classes)")
    
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
                        c2a_class = int(parts[0])
                        
                        # MAP sang VisDrone class
                        visdrone_class = self.C2A_TO_VISDRONE.get(c2a_class, 1)  # Default: pedestrian
                        
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
    
    # Test với đường dẫn thực tế
    try:
        dataset = C2ADataset("c2a/train", transforms=transform)
        
        print(f"\n📊 Dataset Info:")
        print(f"   Total images: {len(dataset)}")
        
        # Test một sample
        if len(dataset) > 0:
            img, target = dataset[0]
            print(f"\n🖼️  Sample 0:")
            print(f"   Image shape: {img.shape}")
            print(f"   Boxes: {target['boxes'].shape}")
            print(f"   Labels: {target['labels']}")
            if len(target['labels']) > 0:
                print(f"   Label range: {target['labels'].min()}-{target['labels'].max()}")
        else:
            print("⚠️  Dataset trống!")
            
    except Exception as e:
        print(f"❌ Lỗi khi test dataset: {e}")
        print(f"ℹ️  Hãy kiểm tra đường dẫn c2a/train/images và c2a/train/labels")
