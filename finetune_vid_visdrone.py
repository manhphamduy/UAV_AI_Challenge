# file: finetune_vid_visdrone.py

import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset_visdrone_vid import VisDroneVideoDataset
# Giả sử file evaluate.py tồn tại và hoạt động đúng
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
# ==== SỬA LỖI 1: TỰ ĐỘNG XÓA CACHE CŨ ====
# ======================================================================
# Điều này đảm bảo rằng mọi thay đổi trong Dataset sẽ được áp dụng.
train_cache_path = os.path.join(train_path, "annotations_cache.pkl")
val_cache_path = os.path.join(val_path, "annotations_cache.pkl")
if os.path.exists(train_cache_path):
    os.remove(train_cache_path)
    print(f"🧹 Removed old train cache: {train_cache_path}")
if os.path.exists(val_cache_path):
    os.remove(val_cache_path)
    print(f"🧹 Removed old validation cache: {val_cache_path}")

# ======================================================================
# ==== AUGMENTATION (CPU PART with Albumentations) ====
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
# ==== DATASET & SỬA LỖI 2: COLLATOR BẢO VỆ ====
# ======================================================================
def collate_fn_robust(batch):
    """
    Collate function tùy chỉnh để lọc ra các sample bị lỗi (trả về None từ Dataset).
    """
    batch = [data for data in batch if data is not None and data[0] is not None]
    if not batch:
        return None, None
    return tuple(zip(*batch))

train_dataset = VisDroneVideoDataset(train_path, transforms=transform_train)
val_dataset = VisDroneVideoDataset(val_path, transforms=transform_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_robust, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_robust, num_workers=2, pin_memory=True)
print("✅ Dataloaders ready.")


# ======================================================================
# ==== MODEL, OPTIMIZER, SCHEDULER ====
# ======================================================================
print("Setting up model, optimizer, and scheduler...")
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
if os.path.exists(pretrained_model_path) and not os.path.exists(checkpoint_path):
    print(f"Loading weights from pre-trained image model: {pretrained_model_path}")
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
model.to(device)
if gpu_count > 1:
    model = torch.nn.DataParallel(model, device_ids=list(range(gpu_count)))
backbone_params = [p for name, p in model.named_parameters() if 'backbone' in name and p.requires_grad]
head_params = [p for name, p in model.named_parameters() if 'backbone' not in name and p.requires_grad]
param_groups = [{'params': backbone_params, 'lr': LR_BACKBONE}, {'params': head_params, 'lr': LR_HEAD}]
optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, eta_min=1e-6)
print("✅ Model setup complete.")


# ======================================================================
# ==== CHECKPOINT LOADING (Đã sửa từ trước) ====
# ======================================================================
start_epoch = 0
best_map = 0.0
if os.path.exists(checkpoint_path):
    print(f"Resuming training from checkpoint: {checkpoint_path}")
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
# ==== TRAINING LOOP & SỬA LỖI 3: KIỂM TRA BATCH RỖNG ====
# ======================================================================
print(f"\n🔥 === Starting Training ({TOTAL_EPOCHS} Epochs) ===")
for epoch in range(start_epoch, TOTAL_EPOCHS):
    model.train()
    total_loss = 0.0
    batches_processed = 0
    current_lr = optimizer.param_groups[1]['lr']
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS} (LR={current_lr:.1e})")

    for images, targets in progress_bar:
        # Kiểm tra xem collator có trả về batch rỗng không (do lọc hết sample lỗi)
        if images is None or not images:
            continue

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        if gpu_count > 1:
            losses = losses.mean()
        if not torch.isfinite(losses):
            print(f"Warning: Found non-finite loss, skipping batch.")
            continue
        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()
        
        total_loss += losses.item()
        batches_processed += 1
        progress_bar.set_postfix(loss=f"{losses.item():.4f}")
    
    # Tránh lỗi chia cho 0 nếu tất cả các batch đều bị lỗi
    avg_loss = total_loss / batches_processed if batches_processed > 0 else 0.0
    print(f"📉 Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")
    
    print(f"📊 Evaluating...")
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