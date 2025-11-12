# file: finetune_vid_visdrone.py (DEBUG VERSION)

import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset_visdrone_vid import VisDroneVideoDataset
from evaluate import evaluate_model

# ======================================================================
# ==== CONFIG ====
# ======================================================================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
gpu_count = torch.cuda.device_count()
print(f"Using device: {device}, Found {gpu_count} GPUs.")
num_classes = 12
IMG_SIZE = 640
TOTAL_EPOCHS = 40
batch_size = 4 * gpu_count if gpu_count > 0 else 4
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
# ==== AUTO-DELETE CACHE ====
# ======================================================================
train_cache_path = os.path.join(train_path, "annotations_cache.pkl")
val_cache_path = os.path.join(val_path, "annotations_cache.pkl")
if os.path.exists(train_cache_path):
    os.remove(train_cache_path)
    print(f"🧹 Removed old train cache: {train_cache_path}")
if os.path.exists(val_cache_path):
    os.remove(val_cache_path)
    print(f"🧹 Removed old validation cache: {val_cache_path}")

# ======================================================================
# ==== AUGMENTATION ====
# ======================================================================
print("Setting up Albumentations pipelines...")
bbox_params = A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.1)
transform_train = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, p=0.8),
    A.ToFloat(max_value=255.0),
    ToTensorV2(),
], bbox_params=bbox_params)
transform_val = A.Compose([
    A.Resize(height=IMG_SIZE, width=IMG_SIZE),
    A.ToFloat(max_value=255.0),
    ToTensorV2(),
], bbox_params=bbox_params)

# ======================================================================
# ==== DATASET & COLLATOR ====
# ======================================================================
def collate_fn_robust(batch):
    batch = [data for data in batch if data is not None and data[0] is not None]
    if not batch:
        return None, None
    return tuple(zip(*batch))

train_dataset = VisDroneVideoDataset(train_path, transforms=transform_train)
val_dataset = VisDroneVideoDataset(val_path, transforms=transform_val)

# SỬA LỖI QUAN TRỌNG: Đặt num_workers=0 để debug
NUM_WORKERS_DEBUG = 0
print(f"🔥🔥🔥 DEBUG MODE: Setting num_workers = {NUM_WORKERS_DEBUG} 🔥🔥🔥")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_robust, num_workers=NUM_WORKERS_DEBUG, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_robust, num_workers=NUM_WORKERS_DEBUG, pin_memory=True)
print("✅ Dataloaders ready.")


# ======================================================================
# ==== MODEL, OPTIMIZER, SCHEDULER ====
# ======================================================================
print("Setting up model, optimizer, and scheduler...")
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
if os.path.exists(pretrained_model_path) and not os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
model.to(device)
if gpu_count > 1:
    model = torch.nn.DataParallel(model, device_ids=list(range(gpu_count)))
param_groups = [{'params': [p for name, p in model.named_parameters() if 'backbone' in name and p.requires_grad], 'lr': LR_BACKBONE},
                {'params': [p for name, p in model.named_parameters() if 'backbone' not in name and p.requires_grad], 'lr': LR_HEAD}]
optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6)
print("✅ Model setup complete.")


# ======================================================================
# ==== CHECKPOINT LOADING ====
# ======================================================================
start_epoch = 0
best_map = 0.0
if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_to_load = model.module if gpu_count > 1 else model
    model_to_load.load_state_dict(ckpt['model_state'])
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
    batches_processed = 0
    current_lr = optimizer.param_groups[1]['lr']
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS} (LR={current_lr:.1e})")

    for i, (images, targets) in enumerate(progress_bar):
        if images is None or not images:
            continue

        # SỬA LỖI MẠNH NHẤT: Thêm câu lệnh ASSERT để kiểm tra từng ảnh
        for img_idx, img in enumerate(images):
            image_id_tensor = targets[img_idx]['image_id']
            original_dataset_idx = image_id_tensor.item()
            img_path = train_dataset.samples[original_dataset_idx]
            
            # Câu lệnh này sẽ dừng chương trình ngay lập tức nếu tìm thấy ảnh không phải 3 kênh
            assert img.shape[0] == 3, \
                f"FATAL: Image at index {original_dataset_idx} ({img_path}) has {img.shape[0]} channels, not 3. Shape: {img.shape}"

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
        loss_dict = model(images, targets)
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
        batches_processed += 1
        progress_bar.set_postfix(loss=f"{losses.item():.4f}")
    
    avg_loss = total_loss / batches_processed if batches_processed > 0 else 0.0
    print(f"📉 Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")
    
    print(f"📊 Evaluating...")
    mAP = evaluate_model(model, val_loader, device) 
    print(f"📊 Epoch {epoch+1} - Validation mAP: {mAP:.4f}")
    
    scheduler.step()
    
    if mAP > best_map:
        best_map = mAP
        torch.save(model.module.state_dict() if gpu_count > 1 else model.state_dict(), vid_model_path)
        print(f"🌟 New best model saved (mAP={best_map:.4f})")
    
    torch.save({
        'epoch': epoch,
        'model_state': model.module.state_dict() if gpu_count > 1 else model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_map': best_map,
    }, checkpoint_path)
    print(f"💾 Checkpoint saved for epoch {epoch+1}")

print(f"\n🎉 Training complete! Best Validation mAP = {best_map:.4f}")