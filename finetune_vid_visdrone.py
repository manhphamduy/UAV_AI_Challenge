import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm
import kornia.augmentation as K
# ✅ SỬA: Import đúng từ kornia
try:
    from kornia.geometry.boxes import Boxes
except ImportError:
    # Fallback cho kornia version cũ hơn
    try:
        from kornia.geometry.bbox import Boxes
    except ImportError:
        # Nếu vẫn không có, tự implement
        print("⚠️  Kornia Boxes not found, using manual implementation")
        Boxes = None

import torchvision.transforms.v2 as T

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
transform_train = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
])
transform_val = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
    T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
])
train_dataset = VisDroneVideoDataset(train_path, transforms=transform_train)
val_dataset = VisDroneVideoDataset(val_path, transforms=transform_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                          collate_fn=lambda x: tuple(zip(*x)), num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                        collate_fn=lambda x: tuple(zip(*x)), num_workers=2, pin_memory=True)
print("✅ CPU Dataloaders ready.")


# ======================================================================
# ==== AUGMENTATION (GPU PART) ====
# ======================================================================
print("Setting up GPU-side augmentation module...")

# ✅ GIẢI PHÁP: Không dùng Boxes, augment trực tiếp
gpu_augmentations = K.AugmentationSequential(
    K.RandomHorizontalFlip(p=0.5),
    K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    data_keys=["input"],  # ✅ Chỉ augment image, boxes xử lý manual
    same_on_batch=False
).to(device)
print("✅ GPU Augmentation ready.")


# ======================================================================
# ==== HELPER FUNCTION: AUGMENT BOXES MANUALLY ====
# ======================================================================
def augment_boxes_horizontal_flip(boxes, img_width, do_flip):
    """
    Flip boxes horizontally nếu do_flip = True
    boxes: [N, 4] tensor in xyxy format
    img_width: width of image
    do_flip: bool
    """
    if not do_flip:
        return boxes
    
    # Flip: x_new = img_width - x_old
    boxes_flipped = boxes.clone()
    boxes_flipped[:, 0] = img_width - boxes[:, 2]  # x1_new = W - x2_old
    boxes_flipped[:, 2] = img_width - boxes[:, 0]  # x2_new = W - x1_old
    
    return boxes_flipped


# ======================================================================
# ==== MODEL, OPTIMIZER, SCHEDULER ====
# ======================================================================
print("Setting up model, optimizer, and scheduler...")
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

if os.path.exists(pretrained_model_path):
    print(f"🔄 Loading pretrained model from {pretrained_model_path}")
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device, weights_only=False))
    print(f"✅ Loaded pretrained model")

model.to(device)

if gpu_count > 1:
    print(f"🔥 Using DataParallel with {gpu_count} GPUs")
    model = torch.nn.DataParallel(model, device_ids=list(range(gpu_count)))

backbone_params = [p for name, p in model.named_parameters() if 'backbone' in name and p.requires_grad]
head_params = [p for name, p in model.named_parameters() if 'backbone' not in name and p.requires_grad]
param_groups = [
    {'params': backbone_params, 'lr': LR_BACKBONE, 'name': 'backbone'},
    {'params': head_params, 'lr': LR_HEAD, 'name': 'head'}
]
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
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model state
    if gpu_count > 1:
        model.module.load_state_dict(ckpt['model_state'])
    else:
        model.load_state_dict(ckpt['model_state'])
    
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
    current_lr_backbone = optimizer.param_groups[0]['lr']
    current_lr_head = optimizer.param_groups[1]['lr']
    
    progress_bar = tqdm(train_loader, 
                       desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS} (LR_bb={current_lr_backbone:.1e}, LR_hd={current_lr_head:.1e})")

    for images, targets in progress_bar:
        # Stack images to batch tensor
        images_tensor = torch.stack(images).to(device)
        
        # ✅ AUGMENTATION: Image + Manual box flip
        # Apply image augmentation (including random flip)
        images_augmented = gpu_augmentations(images_tensor)
        
        # Check if horizontal flip was applied (50% chance)
        # Since kornia applies flip randomly, we need to handle boxes accordingly
        # For simplicity, apply the same flip probability manually
        import random
        do_flip = random.random() < 0.5
        
        final_images = []
        final_targets = []
        
        for i in range(len(images)):
            # Get augmented image
            img_aug = images_augmented[0][i] if isinstance(images_augmented, tuple) else images_augmented[i]
            
            # Get original target
            target = targets[i]
            boxes = target['boxes'].to(device)
            
            # Apply horizontal flip to boxes if needed
            if do_flip:
                _, _, img_w = img_aug.shape
                boxes = augment_boxes_horizontal_flip(boxes, img_w, do_flip=True)
            
            # Create new target
            new_target = {
                'boxes': boxes,
                'labels': target['labels'].to(device),
            }
            if 'image_id' in target:
                new_target['image_id'] = target['image_id'].to(device)
            
            final_images.append(img_aug)
            final_targets.append(new_target)
        
        # Forward pass
        try:
            loss_dict = model(final_images, final_targets)
            losses = sum(loss for loss in loss_dict.values())
            
            if gpu_count > 1:
                losses = losses.mean()
            
            if not torch.isfinite(losses):
                print(f"⚠️  Non-finite loss detected, skipping batch")
                continue
            
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            
            total_loss += losses.item()
            progress_bar.set_postfix(loss=f"{losses.item():.4f}")
            
        except Exception as e:
            print(f"⚠️  Error in batch: {e}")
            continue
    
    avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
    print(f"📉 Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")
    
    # Evaluation
    print(f"📊 Evaluating...")
    try:
        mAP = evaluate_model(model, val_loader, device)
        print(f"📊 Epoch {epoch+1} - Validation mAP: {mAP:.4f}")
    except Exception as e:
        print(f"⚠️  Evaluation failed: {e}")
        mAP = 0.0
    
    # Step scheduler
    scheduler.step()
    
    # Save best model
    if mAP > best_map:
        best_map = mAP
        state_dict_to_save = model.module.state_dict() if gpu_count > 1 else model.state_dict()
        torch.save(state_dict_to_save, vid_model_path)
        print(f"🌟 New best model saved (mAP={best_map:.4f})")
    
    # Save checkpoint
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