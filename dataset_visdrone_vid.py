import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict
from tqdm import tqdm
import pickle # Sử dụng pickle hoặc torch.save/load để cache

class VisDroneVideoDataset(Dataset):
    """
    Dataset class cho bộ dữ liệu VisDrone Video (VID).
    Phiên bản được cải tiến để:
    1. Hỗ trợ torchvision.transforms.v2 cho data augmentation.
    2. Hoàn thiện target dictionary (thêm area, image_id, iscrowd).
    3. Thêm cơ chế cache để tăng tốc độ khởi tạo.
    """
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.seq_dir = os.path.join(root_dir, "sequences")
        self.ann_dir = os.path.join(root_dir, "annotations")
        
        # Đường dẫn file cache
        cache_path = os.path.join(root_dir, "annotations_cache.pkl")

        # --- CƠ CHẾ CACHE ---
        if os.path.exists(cache_path):
            print(f"Loading annotations from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                self.annotations = cached_data['annotations']
                self.samples = cached_data['samples']
            print(f"✅ Loaded {len(self.samples)} images from cache.")
        else:
            print("Cache not found. Processing annotations from scratch...")
            self.annotations = self._process_annotations()
            self.samples = list(sorted(self.annotations.keys()))
    
            # ---> ĐIỂM SỬA LỖI QUAN TRỌNG <---
            # Chuyển defaultdict thành dict thông thường trước khi lưu
            annotations_to_save = dict(self.annotations)
    
            print(f"Saving annotations to cache: {cache_path}")
            with open(cache_path, 'wb') as f:
            # Lưu dict thông thường, không phải defaultdict
                pickle.dump({'annotations': annotations_to_save, 'samples': self.samples}, f)
            print("✅ Cache saved.")

        print(f"✅ Found {len(self.samples)} unique images with annotations.")

    def _process_annotations(self):
        annotations = defaultdict(lambda: {'boxes': [], 'labels': []})
        seq_names = sorted(os.listdir(self.seq_dir))
        
        for seq_name in tqdm(seq_names, desc="Processing annotation files"):
            ann_path = os.path.join(self.ann_dir, seq_name + ".txt")
            if not os.path.exists(ann_path):
                continue
            
            with open(ann_path) as f:
                lines = f.readlines()
            
            for line in lines:
                vals = line.strip().split(',')
                frame_id = int(vals[0])
                x, y, w, h = map(float, vals[2:6])
                cls = int(vals[7])

                if cls == 0 or w <= 0 or h <= 0:
                    continue
                
                img_path = os.path.join(self.seq_dir, seq_name, f"{frame_id:07d}.jpg")
                if os.path.exists(img_path):
                    box = [x, y, x + w, y + h]
                    annotations[img_path]['boxes'].append(box)
                    annotations[img_path]['labels'].append(cls)
        return annotations

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        ann = self.annotations[img_path]
        
        img = Image.open(img_path).convert("RGB")
        
        boxes = torch.as_tensor(ann['boxes'], dtype=torch.float32)
        labels = torch.as_tensor(ann['labels'], dtype=torch.int64)

        # --- CẢI TIẾN: HOÀN THIỆN TARGET DICTIONARY ---
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        
        # Tính diện tích
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        target["area"] = area
        
        # iscrowd
        target["iscrowd"] = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        # --- CẢI TIẾN QUAN TRỌNG NHẤT: ÁP DỤNG TRANSFORMS V2 ---
        if self.transforms:
            # transforms.v2 sẽ tự động áp dụng biến đổi cho cả ảnh và target
            img, target = self.transforms(img, target)
            
        return img, target

    def __len__(self):
        return len(self.samples)