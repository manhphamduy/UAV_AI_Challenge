# file: finetune_vid_visdrone_ddp.py

import os
import torch
import torchvision
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from dataset_visdrone_vid import VisDroneVideoDataset
from evaluate import evaluate_model # Đảm bảo hàm này có thể xử lý model DDP

# ======================================================================
# ==== DDP UTILS ====
# ======================================================================
def setup(rank, world_size):
    """Khởi tạo process group cho DDP."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Dọn dẹp process group."""
    dist.destroy_process_group()

def is_main_process():
    """Kiểm tra xem đây có phải là tiến trình chính (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0

# ======================================================================
# ==== MAIN TRAINING FUNCTION ====
# ======================================================================
def main_worker(rank, world_size, args):
    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)

    # ==== CONFIG ====
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device) # Gán GPU cho tiến trình này

    # ==== AUGMENTATION ====
    bbox_params = A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.1)
    transform_train = A.Compose([
        A.Resize(height=args['IMG_SIZE'], width=args['IMG_SIZE']),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, p=0.8),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ], bbox_params=bbox_params)
    transform_val = A.Compose([
        A.Resize(height=args['IMG_SIZE'], width=args['IMG_SIZE']),
        A.ToFloat(max_value=255.0),
        ToTensorV2(),
    ], bbox_params=bbox_params)

    # ==== DATASET & SAMPLER ====
    if is_main_process():
        # Xóa cache chỉ một lần bởi tiến trình chính
        train_cache_path = os.path.join(args['train_path'], "annotations_cache.pkl")
        val_cache_path = os.path.join(args['val_path'], "annotations_cache.pkl")
        if os.path.exists(train_cache_path):
            os.remove(train_cache_path)
        if os.path.exists(val_cache_path):
            os.remove(val_cache_path)
    dist.barrier() # Đảm bảo rank 0 xóa xong cache trước khi các rank khác tiếp tục

    train_dataset = VisDroneVideoDataset(args['train_path'], transforms=transform_train)
    val_dataset = VisDroneVideoDataset(args['val_path'], transforms=transform_val)
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # collate_fn không đổi
    collate_fn = lambda x: tuple(zip(*x))
    
    train_loader = DataLoader(train_dataset, batch_size=args['batch_size_per_gpu'], sampler=train_sampler, collate_fn=collate_fn, num_workers=args['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args['batch_size_per_gpu'], sampler=val_sampler, collate_fn=collate_fn, num_workers=args['num_workers'], pin_memory=True)
    if is_main_process():
        print("✅ Dataloaders with DistributedSampler ready.")

    # ==== MODEL, OPTIMIZER, SCHEDULER ====
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, args['num_classes'])
    
    # Load state_dict trước khi bọc DDP
    if os.path.exists(args['pretrained_model_path']) and not os.path.exists(args['checkpoint_path']):
        if is_main_process():
            print(f"Loading weights from pre-trained image model: {args['pretrained_model_path']}")
        model.load_state_dict(torch.load(args['pretrained_model_path'], map_location='cpu'))

    model.to(device)
    model = DDP(model, device_ids=[rank])
    
    param_groups = [{'params': [p for name, p in model.named_parameters() if 'backbone' in name and p.requires_grad], 'lr': args['LR_BACKBONE']},
                    {'params': [p for name, p in model.named_parameters() if 'backbone' not in name and p.requires_grad], 'lr': args['LR_HEAD']}]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args['WEIGHT_DECAY'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['TOTAL_EPOCHS'], eta_min=1e-6)
    if is_main_process():
        print("✅ Model with DDP, optimizer, and scheduler ready.")

    # ==== CHECKPOINT LOADING ====
    start_epoch = 0
    best_map = 0.0
    if os.path.exists(args['checkpoint_path']):
        # Load checkpoint trên CPU để tránh xung đột GPU, sau đó map sang device của từng rank
        ckpt = torch.load(args['checkpoint_path'], map_location='cpu')
        model.module.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        best_map = ckpt.get('best_map', 0.0)
        if is_main_process():
            print(f"✅ Resumed from epoch {start_epoch}, best_map={best_map:.4f}")
    elif is_main_process():
        print("🚀 Starting training from scratch.")

    # ==== TRAINING LOOP ====
    if is_main_process():
        print(f"\n🔥 === Starting Training ({args['TOTAL_EPOCHS']} Epochs) on {world_size} GPUs ===")
    
    for epoch in range(start_epoch, args['TOTAL_EPOCHS']):
        train_sampler.set_epoch(epoch) # Quan trọng! Đảm bảo dữ liệu được xáo trộn mỗi epoch
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args['TOTAL_EPOCHS']}", disable=not is_main_process())
        for images, targets in progress_bar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            if not torch.isfinite(losses):
                print(f"Rank {rank}: Warning - Found non-finite loss, skipping batch.")
                continue

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args['GRADIENT_CLIP_NORM'])
            optimizer.step()
            
            total_loss += losses.item()
            if is_main_process():
                progress_bar.set_postfix(loss=f"{losses.item():.4f}")
        
        # Đồng bộ loss từ tất cả các GPU
        avg_loss_tensor = torch.tensor(total_loss / len(train_loader)).to(device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
        
        if is_main_process():
            print(f"📉 Epoch {epoch+1} - Train Loss: {avg_loss_tensor.item():.4f}")
            print(f"📊 Evaluating...")

        # Đánh giá chỉ nên được thực hiện trên một rank để tránh tính toán thừa
        # Hàm evaluate_model cần được điều chỉnh để xử lý model DDP
        mAP = evaluate_model(model, val_loader, device) # Lưu ý: val_loader giờ cũng có sampler
        
        if is_main_process():
            print(f"📊 Epoch {epoch+1} - Validation mAP: {mAP:.4f}")
        
        scheduler.step()
        
        # Lưu model và checkpoint chỉ từ tiến trình chính
        if is_main_process():
            if mAP > best_map:
                best_map = mAP
                torch.save(model.module.state_dict(), args['vid_model_path'])
                print(f"🌟 New best model saved (mAP={best_map:.4f})")
            
            torch.save({
                'epoch': epoch,
                'model_state': model.module.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_map': best_map,
            }, args['checkpoint_path'])
            print(f"💾 Checkpoint saved for epoch {epoch+1}")
            
    cleanup()


if __name__ == '__main__':
    # Các tham số training
    args = {
        'num_classes': 12,
        'IMG_SIZE': 640,
        'TOTAL_EPOCHS': 40,
        'batch_size_per_gpu': 4, # Batch size trên mỗi GPU
        'num_workers': 2,
        'LR_HEAD': 1e-4,
        'LR_BACKBONE': 1e-5,
        'WEIGHT_DECAY': 1e-4,
        'GRADIENT_CLIP_NORM': 1.0,
        'train_path': "data/VisDrone2019-VID-train",
        'val_path': "data/VisDrone2019-VID-val",
        'pretrained_model_path': "models/img_best_model.pth",
        'vid_model_path': "models/vid_best_model.pth",
        'checkpoint_path': "models/vid_checkpoint_v2.pth",
    }
    
    world_size = torch.cuda.device_count()
    if world_size > 1:
        # Sử dụng torch.multiprocessing.spawn để khởi chạy các tiến trình DDP
        mp.spawn(main_worker,
                 args=(world_size, args),
                 nprocs=world_size,
                 join=True)
    else:
        # Chạy trên 1 GPU hoặc CPU mà không cần DDP
        main_worker(0, 1, args)