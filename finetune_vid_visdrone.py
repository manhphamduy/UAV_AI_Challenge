import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import kornia.augmentation as K
# <--- SỬA ĐỔI: Sử dụng lại transforms.v2
import torchvision.transforms.v2 as T

from dataset_visdrone_vid import VisDroneVideoDataset
# Giả sử evaluate_model đã được sửa để hoạt động với DataParallel
from evaluate import evaluate_model 

# ======================================================================
# ==== CONFIG ====
# ======================================================================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
gpu_count = torch.cuda.device_count()
print(f"Using device: {device}, Found {gpu_count} GPUs.")
# ... (Các config khác giữ nguyên) ...
num_classes = 12
IMG_SIZE = 640
TOTAL_EPOCHS = 40
batch_size = 4
LR_HEAD = 1e-4
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0
train_path = "data/VisDrone2019-VID-train"
val_path = "data/VisDrone2019-VID-val"
pretrained_model_path = "models/img_best_model.pth"
vid_model_path = "models/vid_best_model.pth"
checkpoint_path = "models/vid_checkpoint_v2.pth"

# ======================================================================
# ==== DATASET & AUGMENTATION (CPU PART) ====
# ======================================================================
print("Setting up CPU-side dataloaders and transforms...")
# <--- SỬA ĐỔI: Đưa Resize trở lại CPU transform
transform_train = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Resize((IMG_SIZE, IMG_SIZE), antialias=True), # Resize ảnh và bbox trên CPU
])

transform_val = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
])

train_dataset = VisDroneVideoDataset(train_path, transforms=transform_train)
val_dataset = VisDroneVideoDataset(val_path, transforms=transform_val)
# ... DataLoader giữ nguyên ...
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)), num_workers=2, pin_memory=True)
print("✅ CPU Dataloaders ready.")


# ======================================================================
# ==== AUGMENTATION (GPU PART) ====
# ======================================================================
print("Setting up GPU-side augmentation module...")
# <--- SỬA ĐỔI: Bỏ Resize khỏi Kornia, chỉ giữ lại các phép khác
gpu_augmentations = nn.Sequential(
    K.RandomHorizontalFlip(p=0.5),
    # K.ColorJitter(...) # Bạn vẫn có thể giữ các phép nặng ở đây
).to(device)
print("✅ GPU Augmentation ready.")


# ======================================================================
# ==== MODEL, OPTIMIZER, SCHEDULER ====
# ======================================================================
# ... (Phần này giữ nguyên không đổi) ...
print("Setting up model, optimizer, and scheduler...")
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
if os.path.exists(pretrained_model_path):
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
model.to(device)
if gpu_count > 1:
    model = nn.DataParallel(model, device_ids=list(range(gpu_count)))
backbone_params = [p for name, p in model.named_parameters() if 'backbone' in name and p.requires_grad]
head_params = [p for name, p in model.named_parameters() if 'backbone' not in name and p.requires_grad]
param_groups = [{'params': backbone_params, 'lr': LR_BACKBONE}, {'params': head_params, 'lr': LR_HEAD}]
optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6)
print("✅ Model setup complete.")


# ======================================================================
# ==== CHECKPOINT LOADING ====
# ======================================================================
# ... (Phần này giữ nguyên không đổi) ...
start_epoch = 0
best_map = 0.0
if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    start_epoch = ckpt['epoch'] + 1
    best_map = ckpt.get('best_map', 0.0)
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
    current_lr = optimizer.param_groups[1]['lr']
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS} (LR={current_lr:.1e})")

    for images, targets in progress_bar:
        # <--- SỬA ĐỔI: Luồng xử lý đơn giản hơn rất nhiều
        # 1. Stack ảnh (đã được resize trên CPU) thành một batch tensor
        images_tensor = torch.stack(images).to(device)
        
        # 2. Thực hiện augmentation còn lại trên GPU
        images_augmented, transform_matrix = gpu_augmentations(images_tensor, return_transform=True)
        
        # 3. Chuyển targets lên GPU và cập nhật bounding box
        final_images = []
        final_targets = []
        for i in range(len(images)):
            target = targets[i]
            boxes = target['boxes']
            
            # Kornia có thể biến đổi bounding box bằng ma trận transform
            # K.transform_bbox cần boxes ở định dạng (N, 4, 2)
            boxes_corners = K.geometry.bbox_to_corners(boxes)
            boxes_corners_aug = K.transform_points(transform_matrix[i].unsqueeze(0), boxes_corners)
            boxes_aug = K.geometry.corners_to_bbox(boxes_corners_aug)

            new_target = {k: v.to(device) for k, v in target.items()}
            new_target['boxes'] = boxes_aug.to(device)
            final_targets.append(new_target)
            final_images.append(images_augmented[i])
            
        loss_dict = model(final_images, final_targets)
        # ... (Phần còn lại của training loop giữ nguyên) ...
        losses = sum(loss for loss in loss_dict.values())
        if gpu_count > 1:
            losses = losses.mean()
        if not torch.isfinite(losses):
            continue
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()
        total_loss += losses.item()
        progress_bar.set_postfix(loss=f"{losses.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    print(f"📉 Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")
    
    print(f"📊 Evaluating...")
    # Cập nhật evaluate_model để không cần truyền gpu_transform nữa
    mAP = evaluate_model(model, val_loader, device) 
    print(f"📊 Epoch {epoch+1} - Validation mAP: {mAP:.4f}")
    
    scheduler.step()
    
    if mAP > best_map:
        best_map = mAP
        state_dict_to_save = model.module.state_dict() if gpu_count > 1 else model.state_dict()
        torch.save(state_dict_to_save, vid_model_path)
        print(f"🌟 New best model saved (mAP={best_map:.4f})")
    
    state_dict_to_save = model.module.state_dict() if gpu_count > 1 else model.state_dict()
    torch.save({
        'epoch': epoch,
        'model_state': state_dict_to_save,
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_map': best_map,
    }, checkpoint_path)
    print(f"💾 Checkpoint saved for epoch {epoch+1}")

print(f"\n🎉 Training complete! Best Validation mAP = {best_map:.4f}")