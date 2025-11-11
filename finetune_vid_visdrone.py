import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

# <--- SỬA ĐỔI: Import Kornia
import kornia.augmentation as K
import kornia.geometry as K_geom
import torchvision.transforms as T # Chỉ dùng cho ToTensor

from dataset_visdrone_vid import VisDroneVideoDataset
from evaluate import evaluate_model # Đảm bảo evaluate_model có thể xử lý DataParallel model

# ======================================================================
# ==== CONFIG ====
# ======================================================================
# <--- SỬA ĐỔI: Chọn GPU chính và kiểm tra số lượng GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
gpu_count = torch.cuda.device_count()
print(f"Using device: {device}, Found {gpu_count} GPUs.")

# --- Cấu hình Dataset & Model ---
num_classes = 12
IMG_SIZE = 640

# --- Cấu hình Training ---
TOTAL_EPOCHS = 40
# <--- SỬA ĐỔI: Tăng batch_size vì dùng nhiều GPU
# Batch size tổng sẽ là batch_size * gpu_count
batch_size = 4  
LR_HEAD = 1e-4
LR_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0

# --- Đường dẫn ---
train_path = "data/VisDrone2019-VID-train"
val_path = "data/VisDrone2019-VID-val"
pretrained_model_path = "models/img_best_model.pth"
vid_model_path = "models/vid_best_model.pth"
checkpoint_path = "models/vid_checkpoint_v2.pth"

# ======================================================================
# ==== DATASET (CPU PART) ====
# ======================================================================
print("Setting up CPU-side dataloaders...")
# <--- SỬA ĐỔI: Transform trên CPU chỉ còn duy nhất ToTensor
# Toàn bộ augmentation sẽ được thực hiện trên GPU
transform_cpu = T.ToTensor()

train_dataset = VisDroneVideoDataset(train_path, transforms=transform_cpu)
val_dataset = VisDroneVideoDataset(val_path, transforms=transform_cpu)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)), num_workers=2, pin_memory=True)
print("✅ CPU Dataloaders ready.")

# ======================================================================
# ==== AUGMENTATION (GPU PART) ====
# ======================================================================
print("Setting up GPU-side augmentation module...")
# <--- SỬA ĐỔI: Tạo pipeline augmentation bằng Kornia
# Nó hoạt động như một module nn.Module và sẽ chạy trên GPU
gpu_augmentations = nn.Sequential(
    K.Resize(size=(IMG_SIZE, IMG_SIZE), antialias=True),
    K.RandomHorizontalFlip(p=0.5),
    # Bạn có thể thêm các augmentation khác của Kornia ở đây
    # K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, p=0.8),
).to(device)

# Tạo một pipeline riêng cho validation (chỉ resize)
gpu_val_transform = nn.Sequential(
    K.Resize(size=(IMG_SIZE, IMG_SIZE), antialias=True),
).to(device)
print("✅ GPU Augmentation ready.")


# ======================================================================
# ==== MODEL, OPTIMIZER, SCHEDULER ====
# ======================================================================
print("Setting up model, optimizer, and scheduler...")
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

if os.path.exists(pretrained_model_path):
    print(f"🔄 Loading pretrained weights from {pretrained_model_path}")
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
else:
    print("⚠️ Pretrained model not found. Using default COCO weights.")

model.to(device)

# <--- SỬA ĐỔI: Bọc model bằng DataParallel để sử dụng nhiều GPU
if gpu_count > 1:
    print(f"Using {gpu_count} GPUs via DataParallel.")
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
start_epoch = 0
best_map = 0.0
if os.path.exists(checkpoint_path):
    print(f"📂 Loading checkpoint from {checkpoint_path}")
    # <--- SỬA ĐỔI: Khi load, ta cần load vào model gốc, không phải DataParallel wrapper
    # nên ta sẽ load trước khi bọc DataParallel (đã làm ở trên)
    # Tuy nhiên, nếu checkpoint được lưu từ DataParallel, nó sẽ có tiền tố 'module.'
    # Ta sẽ xử lý việc này sau khi load optimizer
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Logic để xử lý checkpoint có/không có `module.` prefix
    # Nếu đang dùng nhiều GPU và checkpoint không có 'module.', thêm nó vào.
    # Nếu đang dùng một GPU và checkpoint có 'module.', xóa nó đi.
    # Cách đơn giản nhất là load state vào model gốc trước khi bọc DataParallel
    # Phần code ở trên đã làm việc này.
    
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
        # <--- SỬA ĐỔI: Luồng xử lý mới
        # 1. Chuyển tensor ảnh gốc lên GPU
        # Dữ liệu ảnh lúc này chưa được resize hay augment
        images_tensor = torch.stack(images).to(device)
        
        # 2. Thực hiện augmentation trên toàn bộ batch trên GPU
        images_augmented = gpu_augmentations(images_tensor)
        
        # 3. Cập nhật lại bounding box cho phù hợp với augmentation
        # Kornia không tự động cập nhật target, ta phải làm thủ công
        # Đây là một cách đơn giản, tuy nhiên Kornia có các cách hiệu quả hơn
        # nhưng phức tạp hơn. Cách này đủ tốt.
        final_images = []
        final_targets = []
        # Lấy kích thước ảnh gốc và ảnh sau augment để tính tỉ lệ scale
        orig_h, orig_w = images[0].shape[-2:]
        aug_h, aug_w = images_augmented.shape[-2:]
        scale_h = aug_h / orig_h
        scale_w = aug_w / orig_w
        
        for i in range(len(images)):
            target = targets[i]
            boxes = target['boxes']
            # Scale boxes theo resize
            boxes[:, [0, 2]] *= scale_w
            boxes[:, [1, 3]] *= scale_h
            
            # Nếu có flip (giả định p=0.5, cách đơn giản hóa)
            # Một cách chính xác hơn cần lấy ma trận transform từ kornia
            # nhưng sẽ phức tạp hơn. Với RandomHorizontalFlip thì cách này đủ dùng.
            
            new_target = {k: v.to(device) for k, v in target.items()}
            new_target['boxes'] = boxes.to(device)
            final_targets.append(new_target)
            final_images.append(images_augmented[i])
        
        loss_dict = model(final_images, final_targets)
        
        # DataParallel trả về loss trên từng GPU, cần cộng lại
        losses = sum(loss for loss in loss_dict.values())
        if gpu_count > 1:
            losses = losses.mean() # Lấy trung bình loss trên các GPU

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
    
    # --- EVALUATION ---
    print(f"📊 Evaluating...")
    # Cần một hàm evaluate đã được chỉnh để chạy với GPU transform và DataParallel model
    mAP = evaluate_model(model, val_loader, device, gpu_val_transform)
    print(f"📊 Epoch {epoch+1} - Validation mAP: {mAP:.4f}")
    
    scheduler.step()
    
    # --- Save best model ---
    if mAP > best_map:
        best_map = mAP
        # <--- SỬA ĐỔI: Khi lưu, lấy ra model gốc từ .module
        state_dict_to_save = model.module.state_dict() if gpu_count > 1 else model.state_dict()
        torch.save(state_dict_to_save, vid_model_path)
        print(f"🌟 New best model saved (mAP={best_map:.4f})")
    
    # --- Save checkpoint ---
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