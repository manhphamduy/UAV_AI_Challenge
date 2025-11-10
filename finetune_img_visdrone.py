import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from dataset_visdrone_det import VisDroneDetDataset
from evaluate import evaluate_model

# ==== CONFIG ====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 12
batch_size = 2

# Cấu hình 2 giai đoạn
FROZEN_EPOCHS = 3      # Freeze backbone 3 epoch (KHÔNG evaluate)
UNFROZEN_EPOCHS = 10    # Fine-tune toàn bộ 5 epoch (CÓ evaluate)
TOTAL_EPOCHS = FROZEN_EPOCHS + UNFROZEN_EPOCHS  # Tổng 8 epoch

LR_FROZEN = 5e-4       # Learning rate cho giai đoạn freeze
LR_UNFROZEN = 1e-5     # Learning rate cho giai đoạn fine-tune
GRADIENT_CLIP_NORM = 1.0
SCHEDULER_STEP_SIZE = 3
SCHEDULER_GAMMA = 0.1

# Đường dẫn
train_path = "data/VisDrone2019-DET-train"
val_path = "data/VisDrone2019-DET-val"
vid_model_path = "models/best_model.pth"
img_model_path = "models/img_best_model.pth"
checkpoint_path = "models/img_checkpoint.pth"

# ==== Dataset ====
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = VisDroneDetDataset(train_path, transforms=transform)
val_dataset = VisDroneDetDataset(val_path, transforms=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# ==== Model ====
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.roi_heads.nms_thresh = 0.3
model.roi_heads.score_thresh = 0.05

# Load model từ VID
print(f"🔄 Loading VID-trained model from {vid_model_path}")
if os.path.exists(vid_model_path):
    model.load_state_dict(torch.load(vid_model_path, map_location=device, weights_only=False))
    print(f"✅ Successfully loaded VID model")
else:
    print(f"⚠️ VID model not found! Training from scratch")

model.to(device)

# ==== Load checkpoint nếu có ====
start_epoch = 0
best_map = 0.0
ckpt = None

if os.path.exists(checkpoint_path):
    print(f"📂 Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    start_epoch = ckpt['epoch'] + 1
    best_map = ckpt['best_map']
    print(f"✅ Resumed from epoch {start_epoch}, best_map={best_map:.4f}")
else:
    print("🚀 Starting training from scratch")

# ======================================================================
# GIAI ĐOẠN 1: FREEZE BACKBONE (3 epoch) - KHÔNG EVALUATE
# ======================================================================
if start_epoch < FROZEN_EPOCHS:
    print("\n🧊 === GIAI ĐOẠN 1: FREEZE BACKBONE (3 EPOCH) - WARM-UP ===")
    print("ℹ️  Giai đoạn này KHÔNG evaluate để tăng tốc training")
    
    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR_FROZEN)
    
    # Load optimizer state nếu có
    if ckpt and 'optimizer_state' in ckpt and ckpt.get('stage') == 1:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    
    for epoch in range(start_epoch, FROZEN_EPOCHS):
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Stage 1 - Epoch {epoch+1}/{FROZEN_EPOCHS}")
        for images, targets in progress_bar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            total_loss += losses.item()
            progress_bar.set_postfix(loss=f"{losses.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"📉 Stage 1 - Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")
        print(f"⏭️  Skipping evaluation for speed")
        
        # Save checkpoint (KHÔNG evaluate)
        torch.save({
            'epoch': epoch,
            'stage': 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_map': best_map,  # Giữ nguyên best_map từ trước
            'last_loss': avg_loss
        }, checkpoint_path)
        print(f"💾 Checkpoint saved")

# ======================================================================
# GIAI ĐOẠN 2: FINE-TUNE TOÀN BỘ (5 epoch) - CÓ EVALUATE
# ======================================================================
start_epoch_stage2 = max(FROZEN_EPOCHS, start_epoch)

if start_epoch < TOTAL_EPOCHS:
    print("\n🔥 === GIAI ĐOẠN 2: FINE-TUNE TOÀN BỘ (5 EPOCH) - WITH EVALUATION ===")
    
    # Unfreeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_UNFROZEN)
    scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)
    
    # Load optimizer & scheduler state nếu có
    if ckpt and 'optimizer_state' in ckpt and ckpt.get('stage') == 2:
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
    
    for epoch in range(start_epoch_stage2, TOTAL_EPOCHS):
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Stage 2 - Epoch {epoch+1}/{TOTAL_EPOCHS} (LR={scheduler.get_last_lr()[0]:.1e})")
        for images, targets in progress_bar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            
            total_loss += losses.item()
            progress_bar.set_postfix(loss=f"{losses.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"📉 Stage 2 - Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")
        
        # EVALUATE (chỉ ở giai đoạn 2)
        print(f"📊 Evaluating...")
        mAP = evaluate_model(model, val_loader, device)
        print(f"📊 Stage 2 - Epoch {epoch+1} - Validation mAP: {mAP:.4f}")
        
        scheduler.step()
        
        # Save best model
        if mAP > best_map:
            best_map = mAP
            torch.save(model.state_dict(), img_model_path)
            print(f"🌟 Stage 2 - New best model saved (mAP={best_map:.4f})")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'stage': 2,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_map': best_map,
            'last_loss': avg_loss
        }, checkpoint_path)
        print(f"💾 Checkpoint saved")

print(f"\n🎉 Training complete! Best mAP = {best_map:.4f}")
