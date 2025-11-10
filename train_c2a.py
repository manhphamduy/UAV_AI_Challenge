import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from dataset_c2a import C2ADataset
from evaluate import evaluate_model

# ======================================================================
# PHẦN CẤU HÌNH TỐI ƯU
# ======================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 12  # Giữ nguyên 12 classes như VisDrone/SARD
batch_size = 2

# --- Cấu hình cho 2 giai đoạn training ---
FROZEN_EPOCHS = 3
LR_HEAD = 1e-5
UNFROZEN_EPOCHS = 8
LR_FINETUNE = 1e-7
TOTAL_EPOCHS = FROZEN_EPOCHS + UNFROZEN_EPOCHS

# --- Cấu hình cho sự ổn định ---
GRADIENT_CLIP_NORM = 1.0
SCHEDULER_STEP_SIZE = 3
SCHEDULER_GAMMA = 0.1

# --- Đường dẫn cho C2A dataset ---
train_path = "data/c2a/train"  # Thay đổi theo đường dẫn thực tế
val_path = "data/c2a/val"
best_model_path = "models/c2a_best_model.pth"
checkpoint_path = "models/checkpoint_c2a.pth"
pretrained_model = "models/sard_best_model.pth"  # Load từ SARD model

os.makedirs("models", exist_ok=True)

# ======================================================================
# KHỞI TẠO CÁC THÀNH PHẦN
# ======================================================================
transform = transforms.Compose([transforms.ToTensor()])
def collate_fn(batch): return tuple(zip(*batch))

train_dataset = C2ADataset(train_path, transforms=transform)
val_dataset = C2ADataset(val_path, transforms=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Load TOÀN BỘ weights từ SARD model (bao gồm cả head)
if os.path.exists(pretrained_model):
    print(f"🔄 Đang load pretrained weights từ {pretrained_model}")
    pretrained_dict = torch.load(pretrained_model, map_location=device, weights_only=False)
    
    # Load TOÀN BỘ (backbone + head) vì cùng 12 classes
    model.load_state_dict(pretrained_dict)
    print(f"✅ Đã load TOÀN BỘ weights từ SARD model (backbone + head)")
    print(f"ℹ️  Số classes: {num_classes} (giữ nguyên)")
else:
    print(f"⚠️  Không tìm thấy {pretrained_model}")
    print(f"⚠️  Fallback: Load từ VisDrone model")
    fallback_model = "models/best_model.pth"
    if os.path.exists(fallback_model):
        pretrained_dict = torch.load(fallback_model, map_location=device, weights_only=False)
        model.load_state_dict(pretrained_dict)
        print(f"✅ Đã load từ {fallback_model}")
    else:
        print(f"⚠️  Training from scratch với ImageNet weights")
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.to(device)

# ======================================================================
# LOGIC KHÔI PHỤC TỪ CHECKPOINT
# ======================================================================
start_epoch = 0
best_map = 0.0
optimizer = None
scheduler = None
ckpt = None

if os.path.exists(checkpoint_path):
    print(f"📄 Tìm thấy checkpoint! Đang khôi phục từ {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    start_epoch = ckpt['epoch'] + 1
    best_map = ckpt['best_map']
    print(f"✅ Khôi phục thành công. Sẽ bắt đầu từ epoch {start_epoch+1}.")
else:
    print("🚀 Không tìm thấy checkpoint. Bắt đầu training từ đầu.")

# ======================================================================
# GIAI ĐOẠN 1: FREEZE BACKBONE (3 epoch) - KHÔNG EVALUATE
# ======================================================================
if start_epoch < FROZEN_EPOCHS:
    print("\n🧊 === GIAI ĐOẠN 1: FREEZE BACKBONE (3 EPOCH) - WARM-UP ===")
    print("ℹ️  Giai đoạn này KHÔNG evaluate để tăng tốc")
    
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR_HEAD)
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
            optimizer.zero_grad(); losses.backward(); optimizer.step()
            total_loss += losses.item()
            progress_bar.set_postfix(loss=f"{losses.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"📉 Stage 1 - Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")
        print(f"⏭️  Skipping evaluation for speed")

        # Lưu checkpoint
        torch.save({
            'epoch': epoch, 'stage': 1, 'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(), 'best_map': best_map
        }, checkpoint_path)
        print(f"💾 Checkpoint saved")

# ======================================================================
# GIAI ĐOẠN 2: FINE-TUNE TOÀN BỘ (8 epoch) - CÓ EVALUATE
# ======================================================================
start_epoch_stage2 = max(FROZEN_EPOCHS, start_epoch)

if start_epoch < TOTAL_EPOCHS:
    print("\n🔥 === GIAI ĐOẠN 2: FINE-TUNE TOÀN BỘ (8 EPOCH) - WITH EVALUATION ===")
    
    for parameter in model.backbone.parameters():
        parameter.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FINETUNE)
    scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

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
            optimizer.zero_grad(); losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            total_loss += losses.item()
            progress_bar.set_postfix(loss=f"{losses.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"📉 Stage 2 - Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")
        
        print(f"📊 Evaluating...")
        mAP = evaluate_model(model, val_loader, device)
        print(f"📊 Stage 2 - Epoch {epoch+1} - Validation mAP: {mAP:.4f}")
        
        scheduler.step()

        if mAP > best_map:
            best_map = mAP
            torch.save(model.state_dict(), best_model_path)
            print(f"🌟 Epoch {epoch+1} - New best model saved with mAP={best_map:.4f}")

        # Lưu checkpoint
        torch.save({
            'epoch': epoch, 'stage': 2, 'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict(),
            'best_map': best_map
        }, checkpoint_path)
        print(f"💾 Checkpoint saved")

print(f"\n🎉 Quá trình training hoàn tất! Model tốt nhất có mAP = {best_map:.4f}")