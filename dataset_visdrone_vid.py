# file: dataset_visdrone_vid.py (Phiên bản siêu nhẹ cho GPU Augmentation)

import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict
from tqdm import tqdm
import pickle
# <--- SỬA ĐỔI: Chỉ cần transforms cơ bản
import torchvision.transforms as T

class VisDroneVideoDataset(Dataset):
    def __init__(self, root_dir, transforms=None): # Mặc dù có transforms, chúng ta sẽ chỉ dùng ToTensor
        self.root_dir = root_dir
        self.transforms = transforms
        # <--- SỬA ĐỔI: Tạo một transform mặc định nếu không có
        if self.transforms is None:
            self.transforms = T.ToTensor()

        self.seq_dir = os.path.join(root_dir, "sequences")
        self.ann_dir = os.path.join(root_dir, "annotations")
        
        cache_path = os.path.join(root_dir, "annotations_cache.pkl")

        if os.path.exists(cache_path):
            print(f"Loading annotations from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                self.annotations = cached_data['annotations']
                self.samples = cached_data['samples']
            print(f"✅ Loaded {len(self.samples)} images from cache.")
        else:
            print("Cache not found. Processing annotations from scratch...")
            self._create_cache(cache_path)

        print(f"✅ Found {len(self.samples)} unique images with annotations.")

    def _create_cache(self, cache_path):
        self.annotations = self._process_annotations()
        self.samples = list(sorted(self.annotations.keys()))
        annotations_to_save = dict(self.annotations)
        print(f"Saving annotations to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump({'annotations': annotations_to_save, 'samples': self.samples}, f)
        print("✅ Cache saved.")

    def _process_annotations(self):
        annotations = defaultdict(lambda: {'boxes': [], 'labels': []})
        seq_names = sorted(os.listdir(self.seq_dir))
        
        for seq_name in tqdm(seq_names, desc="Processing annotation files"):
            # ... (phần code này giữ nguyên) ...
            ann_path = os.path.join(self.ann_dir, seq_name + ".txt")
            if not os.path.exists(ann_path): continue
            with open(ann_path) as f: lines = f.readlines()
            for line in lines:
                vals = line.strip().split(',')
                frame_id, x, y, w, h, cls = int(vals[0]), float(vals[2]), float(vals[3]), float(vals[4]), float(vals[5]), int(vals[7])
                if cls == 0 or w <= 0 or h <= 0: continue
                img_path = os.path.join(self.seq_dir, seq_name, f"{frame_id:07d}.jpg")
                if os.path.exists(img_path):
                    annotations[img_path]['boxes'].append([x, y, x + w, y + h])
                    annotations[img_path]['labels'].append(cls)
        return annotations

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        ann = self.annotations[img_path]
        
        img = Image.open(img_path).convert("RGB")
        
        boxes = torch.as_tensor(ann['boxes'], dtype=torch.float32)
        labels = torch.as_tensor(ann['labels'], dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        target["area"] = area
        target["iscrowd"] = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        # <--- SỬA ĐỔI: Chỉ áp dụng ToTensor trên CPU
        # img bây giờ là một tensor, nhưng target vẫn giữ nguyên
        img = self.transforms(img)
            
        return img, target

    def __len__(self):
        return len(self.samples)