import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# Cập nhật import cho bộ dữ liệu Video
from dataset_visdrone_vid import VisDroneVideoDataset
from evaluate import evaluate_model

# ======================================================================
# PHẦN CẤU HÌNH TỐI ƯU
# ======================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 12
batch_size = 2

# --- Cấu hình cho 2 giai đoạn training ---
FROZEN_EPOCHS = 3
LR_HEAD = 0.001
UNFROZEN_EPOCHS = 8
LR_FINETUNE = 5e-5
TOTAL_EPOCHS = FROZEN_EPOCHS + UNFROZEN_EPOCHS

# --- Cấu hình cho sự ổn định ---
GRADIENT_CLIP_NORM = 1.0
SCHEDULER_STEP_SIZE = 4
SCHEDULER_GAMMA = 0.1

# --- Đường dẫn (cập nhật cho bộ VID) ---
train_path = "data/VisDrone2019-VID-train"
val_path = "data/VisDrone2019-VID-val"
best_model_path = "models/best_model.pth"
checkpoint_path = "models/checkpoint_visdrone.pth" # <-- Cập nhật checkpoint path

os.makedirs("models", exist_ok=True)
# ======================================================================
# KHỞI TẠO CÁC THÀNH PHẦN
# ======================================================================
transform = transforms.Compose([transforms.ToTensor()])
def collate_fn(batch): return tuple(zip(*batch))

# Cập nhật tên Dataset
train_dataset = VisDroneVideoDataset(train_path, transforms=transform)
val_dataset = VisDroneVideoDataset(val_path, transforms=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

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
cpkt=None

if os.path.exists(checkpoint_path):
    print(f"🔄 Tìm thấy checkpoint! Đang khôi phục từ {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    start_epoch = ckpt['epoch']+1
    best_map = ckpt['best_map']
    print(f"✅ Khôi phục thành công. Sẽ bắt đầu từ epoch {start_epoch+1}.")
else:
    print("🚀 Không tìm thấy checkpoint. Bắt đầu training từ đầu.")

# ======================================================================
# BẮT ĐẦU QUÁ TRÌNH TRAINING
# ======================================================================

# Chạy Giai đoạn 1 nếu chưa hoàn thành
if start_epoch < FROZEN_EPOCHS:
    print("\n--- Bắt đầu/Tiếp tục GIAI ĐOẠN 1: WARM-UP ---")
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR_HEAD)
    if ckpt and 'optimizer_state' in ckpt and ckpt.get('stage') == 1:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    # if os.path.exists(checkpoint_path) and 'optimizer_state' in ckpt and ckpt.get('stage') == 1:
    #     optimizer.load_state_dict(ckpt['optimizer_state'])

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

        # Lưu checkpoint sau mỗi epoch
        torch.save({
            'epoch': epoch, 'stage': 1, 'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(), 'best_map': best_map
        }, checkpoint_path)

# Cập nhật start_epoch cho giai đoạn 2
start_epoch_stage2 = max(FROZEN_EPOCHS, start_epoch)

# Chạy Giai đoạn 2 nếu chưa hoàn thành
if start_epoch < TOTAL_EPOCHS:
    print("\n--- Bắt đầu/Tiếp tục GIAI ĐOẠN 2: FINE-TUNE ---")
    for parameter in model.backbone.parameters():
        parameter.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FINETUNE)
    scheduler = StepLR(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA)

    # if os.path.exists(checkpoint_path) and 'optimizer_state' in ckpt and ckpt.get('stage') == 2:
    #     optimizer.load_state_dict(ckpt['optimizer_state'])
    #     scheduler.load_state_dict(ckpt['scheduler_state'])
    
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
        
        mAP = evaluate_model(model, val_loader, device)
        scheduler.step()

        if mAP > best_map:
            best_map = mAP
            torch.save(model.state_dict(), best_model_path)
            print(f"🌟 Epoch {epoch+1} - New best model saved with mAP={best_map:.4f}")

        # Lưu checkpoint sau mỗi epoch
        torch.save({
            'epoch': epoch, 'stage': 2, 'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict(),
            'best_map': best_map
        }, checkpoint_path)

print(f"\n🏁 Quá trình training hoàn tất! Model tốt nhất có mAP = {best_map:.4f}")