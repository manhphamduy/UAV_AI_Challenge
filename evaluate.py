# file: evaluate.py
# Yêu cầu cài đặt: pip install torchmetrics

import torch
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision

def evaluate_model(model, data_loader, device):
    """
    Đánh giá model object detection trên tập dữ liệu validation.

    Hàm này sẽ:
    1. Xử lý model được bọc trong nn.DataParallel.
    2. Chạy inference trên toàn bộ data_loader.
    3. Sử dụng torchmetrics để tính toán mAP một cách hiệu quả trên GPU.

    Args:
        model (torch.nn.Module): Model cần đánh giá.
        data_loader (DataLoader): DataLoader cho tập validation.
        device (torch.device): Thiết bị để chạy (CPU hoặc GPU).

    Returns:
        float: Giá trị mAP@.50:.95 (tiêu chuẩn COCO).
    """
    
    # 1. Lấy model gốc nếu đang dùng DataParallel
    # Khi gọi model(images), ta vẫn dùng `model` đã được bọc.
    # Nhưng khi cần truy cập các thuộc tính/phương thức gốc, ta dùng `model_eval`.
    if isinstance(model, torch.nn.DataParallel):
        model_eval = model.module
    else:
        model_eval = model
        
    # Chuyển model sang chế độ đánh giá
    model_eval.eval()
    
    # 2. Khởi tạo metric tính toán mAP
    # box_format='xyxy' khớp với output của model Faster R-CNN
    metric = MeanAveragePrecision(box_format='xyxy').to(device)
    
    # Tắt tính toán gradient để tăng tốc và tiết kiệm bộ nhớ
    with torch.no_grad():
        # Vòng lặp qua tập validation
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            
            # Chuyển dữ liệu lên device
            # images là một list các tensor, targets là một list các dict
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 3. Lấy kết quả dự đoán từ model
            # Model sẽ trả về một list các dict, mỗi dict chứa 'boxes', 'labels', 'scores'
            outputs = model(images)
            
            # 4. Cập nhật metric với kết quả dự đoán và ground truth
            # torchmetrics sẽ tự động xử lý toàn bộ việc tính toán phức tạp
            metric.update(outputs, targets)
            
    # 5. Tính toán kết quả mAP cuối cùng
    print("Computing final mAP score...")
    results = metric.compute()
    
    # Trích xuất giá trị mAP chính (IoU threshold từ 0.5 đến 0.95)
    map_score = results['map'].item()
    
    # In ra các chỉ số khác để tham khảo
    print(f"mAP@.50:.95 (primary): {map_score:.4f}")
    print(f"mAP@.50: {results['map_50'].item():.4f}")
    print(f"mAP@.75: {results['map_75'].item():.4f}")
    
    return map_score