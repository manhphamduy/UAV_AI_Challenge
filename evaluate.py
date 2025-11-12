# file: evaluate.py
# Yêu cầu cài đặt: pip install torchmetrics

import torch
import torch.distributed as dist
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision

def is_dist_avail_and_initialized():
    """Kiểm tra xem DDP có được sử dụng và khởi tạo chưa."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def evaluate_model(model, data_loader, device):
    """
    Đánh giá model object detection, tương thích với cả single-GPU, DataParallel và DDP.

    Hàm này sẽ:
    1. Lấy model gốc từ DDP hoặc DataParallel wrapper.
    2. Chạy inference trên data_loader (mỗi GPU sẽ xử lý một phần dữ liệu).
    3. Sử dụng torchmetrics để tính toán mAP. Torchmetrics sẽ tự động đồng bộ
       kết quả từ tất cả các GPU khi gọi .compute() trong môi trường DDP.

    Args:
        model (torch.nn.Module): Model cần đánh giá (có thể là DDP-wrapped).
        data_loader (DataLoader): DataLoader cho tập validation (có thể có DistributedSampler).
        device (torch.device): Thiết bị để chạy (CPU hoặc GPU cụ thể của rank).

    Returns:
        float: Giá trị mAP@.50:.95. Trong môi trường DDP, chỉ rank 0 mới trả về
               giá trị này, các rank khác trả về 0.0.
    """
    
    # 1. Lấy model gốc từ DDP hoặc DataParallel wrapper
    # Khi gọi model(images), ta vẫn dùng `model` đã được bọc.
    if isinstance(model, (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)):
        model_eval = model.module
    else:
        model_eval = model
        
    # Chuyển model sang chế độ đánh giá
    model_eval.eval()
    
    # 2. Khởi tạo metric tính toán mAP
    # sync_on_compute=True (mặc định) sẽ tự động đồng bộ kết quả giữa các tiến trình
    # khi .compute() được gọi trong môi trường DDP.
    metric = MeanAveragePrecision(box_format='xyxy').to(device)
    
    # Xác định xem có nên hiển thị progress bar hay không (chỉ rank 0 mới hiển thị)
    disable_tqdm = is_dist_avail_and_initialized() and dist.get_rank() != 0
    
    # Tắt tính toán gradient
    with torch.no_grad():
        # Vòng lặp qua tập validation
        for images, targets in tqdm(data_loader, desc="Evaluating", disable=disable_tqdm):
            
            # collate_fn trong DataLoader của bạn có thể trả về batch rỗng nếu có sample lỗi
            if images is None or not images:
                continue

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 3. Lấy kết quả dự đoán từ model
            outputs = model(images)
            
            # 4. Cập nhật metric với kết quả của batch hiện tại trên GPU này
            metric.update(outputs, targets)
            
    # 5. Tính toán kết quả mAP cuối cùng
    # Trong môi trường DDP, torchmetrics sẽ tự động thực hiện all_gather để
    # thu thập kết quả từ tất cả các GPU về rank 0 trước khi tính toán.
    # Các rank khác sẽ chỉ đồng bộ và không thực hiện tính toán.
    if not disable_tqdm:
         print("Computing final mAP score (torchmetrics will sync across GPUs)...")
    
    results = metric.compute()
    
    # Chuyển kết quả về CPU và lấy giá trị
    map_score = results['map'].cpu().item()

    # Chỉ tiến trình chính (rank 0) mới in kết quả chi tiết
    if not disable_tqdm:
        print(f"mAP@.50:.95 (primary): {map_score:.4f}")
        print(f"mAP@.50: {results['map_50'].cpu().item():.4f}")
        print(f"mAP@.75: {results['map_75'].cpu().item():.4f}")
    
    return map_score