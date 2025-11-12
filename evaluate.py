# Trong file evaluate.py

import torch
import torch.distributed as dist

# Hàm này để thu thập kết quả từ tất cả các GPU
def gather_results(data):
    world_size = dist.get_world_size()
    if world_size == 1:
        return [data]
    
    output_list = [None for _ in range(world_size)]
    dist.all_gather_object(output_list, data)
    return output_list

def evaluate_model(model, data_loader, device):
    # Lấy model gốc từ DDP wrapper
    model_without_ddp = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    
    # ... (code evaluate của bạn giữ nguyên) ...
    # Nhưng thay vì trả về kết quả ngay, bạn cần thu thập kết quả
    
    # Giả sử bạn có all_predictions và all_targets là list các kết quả
    # ...
    
    # Thu thập kết quả từ tất cả các GPU
    if dist.is_initialized():
        # Giả sử `predictions` là kết quả của rank hiện tại
        gathered_predictions = gather_results(predictions) 
        gathered_targets = gather_results(targets)

        # Chỉ rank 0 mới tính toán mAP cuối cùng
        if dist.get_rank() == 0:
            # Gộp các list kết quả lại
            final_predictions = [item for sublist in gathered_predictions for item in sublist]
            final_targets = [item for sublist in gathered_targets for item in sublist]
            # Tính mAP trên final_predictions và final_targets
            mAP = calculate_map(final_predictions, final_targets) 
            return mAP
        else:
            # Các rank khác không cần trả về gì
            return None 
    else:
        # Trường hợp chạy trên 1 GPU, tính toán như bình thường
        mAP = calculate_map(predictions, targets)
        return mAP