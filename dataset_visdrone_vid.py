import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict
from tqdm import tqdm

class VisDroneVideoDataset(Dataset):
    """
    Dataset class cho bộ dữ liệu VisDrone Video (VID).
    Phiên bản này được tối ưu để gom nhóm các annotations theo từng ảnh,
    giúp quá trình training hiệu quả và chính xác hơn.
    """
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.seq_dir = os.path.join(root_dir, "sequences")
        self.ann_dir = os.path.join(root_dir, "annotations")
        self.seq_names = sorted(os.listdir(self.seq_dir))

        # Sử dụng defaultdict để dễ dàng gom nhóm các bounding box cho mỗi ảnh
        # Cấu trúc: { 'đường_dẫn_ảnh': {'boxes': [[...], ...], 'labels': [..., ...]} }
        self.annotations = defaultdict(lambda: {'boxes': [], 'labels': []})
        
        print(f"Loading video dataset annotations from: {root_dir}")
        
        # Vòng lặp để đọc tất cả các file chú thích và xây dựng cấu trúc dữ liệu
        for seq_name in tqdm(self.seq_names, desc="Processing annotation files"):
            ann_path = os.path.join(self.ann_dir, seq_name + ".txt")
            
            # Bỏ qua nếu sequence không có file chú thích
            if not os.path.exists(ann_path):
                continue
            
            with open(ann_path) as f:
                lines = f.readlines()
            
            for line in lines:
                # Định dạng: <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,...
                vals = line.strip().split(',')
                
                # Chuyển đổi các giá trị cần thiết
                frame_id = int(vals[0])
                # Đọc tọa độ là float để tránh lỗi làm tròn
                x, y, w, h = map(float, vals[2:6]) 
                # Nhãn lớp ở vị trí thứ 7 trong bộ VID
                cls = int(vals[7])

                # Bỏ qua các lớp không quan tâm (0=ignored) hoặc các box không hợp lệ
                if cls == 0 or w <= 0 or h <= 0:
                    continue
                
                # Tạo đường dẫn đầy đủ tới file ảnh
                # Định dạng frame_id được đệm bằng số 0 để có 7 chữ số (ví dụ: 0000001.jpg)
                img_path = os.path.join(self.seq_dir, seq_name, f"{frame_id:07d}.jpg")
                
                # Chỉ thêm vào nếu file ảnh thực sự tồn tại
                if os.path.exists(img_path):
                    # Chuyển đổi box từ [x, y, w, h] sang [x1, y1, x2, y2]
                    box = [x, y, x + w, y + h]
                    self.annotations[img_path]['boxes'].append(box)
                    self.annotations[img_path]['labels'].append(cls)

        # Tạo một danh sách các ảnh duy nhất để có thể truy cập bằng index (idx)
        # Sắp xếp để đảm bảo thứ tự nhất quán
        self.samples = list(sorted(self.annotations.keys()))
        
        print(f"✅ Found {len(self.samples)} unique images with annotations.")

    def __getitem__(self, idx):
        # Lấy đường dẫn ảnh và các chú thích tương ứng
        img_path = self.samples[idx]
        ann = self.annotations[img_path]
        
        # Mở ảnh và chuyển sang định dạng RGB
        img = Image.open(img_path).convert("RGB")
        
        # Chuyển đổi danh sách các box và label thành tensor
        boxes = torch.as_tensor(ann['boxes'], dtype=torch.float32)
        labels = torch.as_tensor(ann['labels'], dtype=torch.int64)
        
        # Tạo target dictionary theo định dạng yêu cầu của torchvision
        target = {"boxes": boxes, "labels": labels}
        
        # Áp dụng transform nếu có
        if self.transforms:
            img = self.transforms(img)
            
        return img, target

    def __len__(self):
        # Độ dài của dataset chính là số lượng ảnh duy nhất có chú thích
        return len(self.samples)