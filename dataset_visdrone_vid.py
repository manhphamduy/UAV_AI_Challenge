# file: dataset_visdrone_vid.py

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
from tqdm import tqdm
import pickle
import cv2

def clip_bboxes(boxes, img_shape):
    """
    Cắt các bounding box để đảm bảo chúng nằm hoàn toàn trong ảnh.
    Args:
        boxes (list or np.array): List các bounding box, format [xmin, ymin, xmax, ymax, ...].
        img_shape (tuple): (height, width) của ảnh.
    Returns:
        np.array: Các bounding box đã được cắt.
    """
    height, width = img_shape
    clipped_boxes = []
    for box in boxes:
        # Tách tọa độ và các thông tin khác (như label)
        coords = box[:4]
        other_info = box[4:]
        
        # Cắt các tọa độ
        xmin, ymin, xmax, ymax = coords
        xmin = max(0, min(xmin, width - 1))
        ymin = max(0, min(ymin, height - 1))
        xmax = max(0, min(xmax, width - 1))
        ymax = max(0, min(ymax, height - 1))
        
        # Chỉ giữ lại box nếu nó vẫn có kích thước hợp lệ (w > 0, h > 0)
        if xmax > xmin and ymax > ymin:
            clipped_boxes.append([xmin, ymin, xmax, ymax] + other_info)
            
    return np.array(clipped_boxes)


class VisDroneVideoDataset(Dataset):
    # ... __init__, _create_cache, _process_annotations không đổi ...
    # Bạn có thể giữ nguyên các hàm này.
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
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
        try:
            img_path = self.samples[idx]
            ann = self.annotations[img_path]
            
            image = cv2.imread(img_path)
            
            if image is None:
                return None, None 
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if len(image.shape) != 3 or image.shape[2] != 3:
                return None, None
            
            # Lấy kích thước ảnh gốc (height, width)
            h, w = image.shape[:2]

            boxes = np.array(ann['boxes'], dtype=np.float32)
            labels = np.array(ann['labels'], dtype=np.int64)
            
            # SỬA LỖI TẠI ĐÂY: Cắt các bounding box để đảm bảo nằm trong ảnh
            if len(boxes) > 0:
                # Cắt các tọa độ
                boxes[:, 0] = np.clip(boxes[:, 0], 0, w) # xmin
                boxes[:, 1] = np.clip(boxes[:, 1], 0, h) # ymin
                boxes[:, 2] = np.clip(boxes[:, 2], 0, w) # xmax
                boxes[:, 3] = np.clip(boxes[:, 3], 0, h) # ymax
                
                # Loại bỏ các box có kích thước bằng 0 sau khi cắt
                valid_indices = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
                boxes = boxes[valid_indices]
                labels = labels[valid_indices]
            
            if self.transforms:
                # Albumentations bây giờ sẽ nhận được dữ liệu hợp lệ
                transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
                image = transformed['image']
                boxes = transformed['bboxes']
                labels = transformed['labels']

            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = torch.tensor([idx])
            
            if boxes.shape[0] > 0:
                area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            else:
                area = torch.tensor([], dtype=torch.float32)
                
            target["area"] = area
            target["iscrowd"] = torch.zeros((boxes.shape[0],), dtype=torch.int64)
                
            return image, target
        except Exception as e:
            print(f"Error processing sample {idx} ({self.samples[idx]}): {e}. Returning None.")
            return None, None

    def __len__(self):
        return len(self.samples)