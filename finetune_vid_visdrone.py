import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.v2 as T # SỬ DỤNG transforms v2
from tqdm import tqdm

# Đảm bảo bạn đang dùng file dataset đã được cải tiến
from dataset_visdrone_vid import VisDroneVideoDataset 
from evaluate import evaluate_model

# ======================================================================
# ==== CONFIG ====
# ======================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Cấu hình Dataset & Model ---
num_classes = 12  # 11 lớp của VisDrone + 1 lớp background
IMG_SIZE = 640    # Kích thước ảnh đầu vào, quan trọng cho vật thể nhỏ

# --- Cấu hình Training ---
TOTAL_EPOCHS = 40       # TĂNG SỐ EPOCHS LÊN ĐÁNG KỂ
batch_size = 2          # Giảm nếu gặp lỗi Out of Memory (OOM) khi tăng IMG_SIZE
LR_HEAD = 1e-4          # Learning rate cho RPN và RoI Heads (cao hơn)
LR_BACKBONE = 1e-5      # Learning rate cho Backbone (thấp hơn 10 lần)
WEIGHT_DECAY = 1e-4     # Sử dụng weight decay với AdamW
GRADIENT_CLIP_NORM = 1.0

# --- Đường dẫn ---
train_path = "data/VisDrone2019-VID-train"
val_path = "data/VisDrone2019-VID-val"
# Model này là kết quả từ việc train trên tập ảnh tĩnh (nếu có)
pretrained_model_path = "models/img_best_model.pth" 
vid_model_path = "models/vid_best_model.pth"
# Đổi tên checkpoint để không ghi đè lên file cũ
checkpoint_path = "models/vid_checkpoint_v2.pth" 

# ======================================================================
# ==== DATASET & AUGMENTATION ====
# ======================================================================
print("Setting up data augmentation and dataloaders...")
# Pipeline transform cho training (bao gồm augmentation)
transform_train = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Resize((IMG_SIZE, IMG_SIZE), antialias=True), # Resize ảnh và bounding box
    T.RandomHorizontalFlip(p=0.5),
    # Có thể thêm ColorJitter để tăng độ khó, nhưng hãy thử không có nó trước
    # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
])

# Pipeline transform cho validation (chỉ resize và chuẩn hóa)
transform_val = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
])

# Khởi tạo Dataset
train_dataset = VisDroneVideoDataset(train_path, transforms=transform_train)
val_dataset = VisDroneVideoDataset(val_path, transforms=transform_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)), num_workers=4, pin_memory=True)
print("✅ Data setup complete.")

# ======================================================================
# ==== MODEL, OPTIMIZER, SCHEDULER ====
# ======================================================================
print("Setting up model, optimizer, and scheduler...")
# Sử dụng model đã pre-train trên COCO
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load model đã được fine-tune trên ảnh tĩnh (nếu có)
if os.path.exists(pretrained_model_path):
    print(f"🔄 Loading pretrained weights from {pretrained_model_path}")
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
else:
    print("⚠️ Pretrained model not found. Using default COCO weights.")

model.to(device)

# --- Phân chia parameters cho Differential Learning Rates ---
backbone_params = [p for name, p in model.named_parameters() if 'backbone' in name and p.requires_grad]
head_params = [p for name, p in model.named_parameters() if 'backbone' not in name and p.requires_grad]

param_groups = [
    {'params': backbone_params, 'lr': LR_BACKBONE},
    {'params': head_params, 'lr': LR_HEAD}
]

# --- Optimizer và Scheduler ---
optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6)
print("✅ Model setup complete.")

# ======================================================================
# ==== CHECKPOINT LOADING ====
# ======================================================================
start_epoch = 0
best_map = 0.0
if os.path.exists(checkpoint_path):
    print(f"📂 Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    start_epoch = ckpt['epoch'] + 1
    best_map = ckpt.get('best_map', 0.0) # Dùng .get để tương thích với checkpoint cũ
    print(f"✅ Resumed from epoch {start_epoch}, best_map={best_map:.4f}")
else:
    print("🚀 Starting training from scratch.")

# ======================================================================
# ==== TRAINING LOOP ====
# ======================================================================
print(f"\n🔥 === Starting Training ({TOTAL_EPOCHS} Epochs) ===")
for epoch in range(start_epoch, TOTAL_EPOCHS):
    model.train()
    total_loss = 0.0
    
    # Lấy LR hiện tại của head để hiển thị
    current_lr = optimizer.param_groups[1]['lr']
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS} (LR={current_lr:.1e})")

    for images, targets in progress_bar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Check for invalid loss
        if not torch.isfinite(losses):
            print(f"WARNING: Non-finite loss detected: {losses.item()}. Skipping batch.")
            continue
            
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()
        
        total_loss += losses.item()
        progress_bar.set_postfix(loss=f"{losses.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    print(f"📉 Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")
    
    # --- EVALUATION (Sau mỗi epoch) ---
    print(f"📊 Evaluating...")
    mAP = evaluate_model(model, val_loader, device)
    print(f"📊 Epoch {epoch+1} - Validation mAP: {mAP:.4f}")
    
    # --- Cập nhật scheduler ---
    scheduler.step()
    
    # --- Save best model ---
    if mAP > best_map:
        best_map = mAP
        torch.save(model.state_dict(), vid_model_path)
        print(f"🌟 New best model saved (mAP={best_map:.4f})")
    
    # --- Save checkpoint ---
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_map': best_map,
    }, checkpoint_path)
    print(f"💾 Checkpoint saved for epoch {epoch+1}")

print(f"\n🎉 Training complete! Best Validation mAP = {best_map:.4f}")