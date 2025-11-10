# evaluate.py (PHIÊN BẢN CUỐI CÙNG - DÙNG PYTHON THUẦN TÚY)

import torch
import numpy as np
from tqdm import tqdm

def compute_iou(box_a, box_b):
    """
    Tính IoU giữa 2 box bằng Python thuần túy.
    An toàn, đơn giản và không phụ thuộc vào các lỗi NumPy.
    """
    # Tọa độ của box A
    x1_a, y1_a, x2_a, y2_a = box_a
    # Tọa độ của box B
    x1_b, y1_b, x2_b, y2_b = box_b

    # Tính tọa độ của vùng giao nhau (intersection)
    x1_inter = max(x1_a, x1_b)
    y1_inter = max(y1_a, y1_b)
    x2_inter = min(x2_a, x2_b)
    y2_inter = min(y2_a, y2_b)

    # Tính diện tích vùng giao nhau
    # max(0, ...) để đảm bảo nếu không có giao nhau thì diện tích là 0
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Tính diện tích của mỗi box
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)

    # Tính diện tích vùng hợp nhất (union)
    union_area = area_a + area_b - inter_area

    # Tính IoU
    iou = inter_area / (union_area + 1e-6)  # Thêm epsilon để tránh chia cho 0
    return iou


def evaluate_model(model, data_loader, device, iou_thresh=0.5):
    model_to_eval = model.module if isinstance(model, torch.nn.DataParallel) else model
    model_to_eval.eval()
    
    aps = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            images, targets = batch
            
            images = list(img.to(device) for img in images)
            
            if isinstance(targets, dict):
                targets = [targets]
            
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model_to_eval(images)

            for output, target in zip(outputs, targets):
                pred_boxes = output['boxes'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                gt_boxes = target['boxes'].cpu().numpy()

                if len(gt_boxes) == 0:
                    aps.append(1.0 if len(pred_boxes) == 0 else 0.0)
                    continue
                if len(pred_boxes) == 0:
                    aps.append(0.0)
                    continue

                sorted_indices = np.argsort(-pred_scores)
                pred_boxes = pred_boxes[sorted_indices]

                iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
                for i, pred_box in enumerate(pred_boxes):
                    for j, gt_box in enumerate(gt_boxes):
                        # Hàm compute_iou mới trả về một số float, không cần .item()
                        iou_matrix[i, j] = compute_iou(pred_box, gt_box)
                
                # Phần còn lại của hàm giữ nguyên
                tp = 0; fp = 0
                precision_list = []; recall_list = []
                gt_detected = np.zeros(len(gt_boxes))

                for i in range(len(pred_boxes)):
                    best_gt_idx = np.argmax(iou_matrix[i, :])
                    best_iou = iou_matrix[i, best_gt_idx]
                    
                    if best_iou >= iou_thresh and not gt_detected[best_gt_idx]:
                        tp += 1; gt_detected[best_gt_idx] = 1
                    else:
                        fp += 1
                    
                    precision = tp / (tp + fp)
                    recall = tp / len(gt_boxes)
                    precision_list.append(precision); recall_list.append(recall)

                ap = 0.0
                for t in np.arange(0., 1.1, 0.1):
                    precisions_at_recall_t = [p for p, r in zip(precision_list, recall_list) if r >= t]
                    p_interp = max(precisions_at_recall_t) if precisions_at_recall_t else 0.0
                    ap += p_interp
                ap /= 11.0
                aps.append(ap)
    
    mAP = np.mean(aps) if aps else 0.0
    print(f"📊 Validation mAP: {mAP:.4f}")
    return mAP