import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# ==== CONFIG ====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model1_path = "models/best_model.pth"
model2_path = "models/img_best_model.pth"

# ==== HÀM LOAD VÀ PHÂN TÍCH MODEL ====
def load_and_analyze_model(model_path, model_name):
    print("\n" + "="*60)
    print(f"📂 PHÂN TÍCH: {model_name}")
    print("="*60)
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Kiểm tra loại checkpoint
        if isinstance(checkpoint, dict):
            print(f"📦 Loại: Dictionary checkpoint")
            print(f"🔑 Keys: {list(checkpoint.keys())}")
            
            # Nếu có metadata
            if 'epoch' in checkpoint:
                print(f"📅 Epoch: {checkpoint['epoch']}")
            if 'best_map' in checkpoint:
                print(f"🎯 Best mAP: {checkpoint['best_map']:.4f}")
            if 'last_loss' in checkpoint:
                print(f"📉 Last Loss: {checkpoint['last_loss']:.4f}")
            if 'stage' in checkpoint:
                print(f"🔢 Stage: {checkpoint['stage']}")
            
            # Load state_dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
            else:
                state_dict = checkpoint
        else:
            print(f"📦 Loại: State dict only")
            state_dict = checkpoint
        
        # Phân tích state_dict
        print(f"\n📊 THỐNG KÊ STATE DICT:")
        print(f"   Tổng số layers: {len(state_dict)}")
        
        # Đếm số parameter
        total_params = 0
        trainable_params = 0
        for key, value in state_dict.items():
            total_params += value.numel()
        
        print(f"   Tổng số parameters: {total_params:,}")
        print(f"   Kích thước file: {torch.cuda.get_device_properties(0).total_memory / 1024**2 if torch.cuda.is_available() else 'N/A'}")
        
        # Hiển thị một số layer quan trọng
        print(f"\n🔍 MỘT SỐ LAYERS QUAN TRỌNG:")
        important_keys = [
            'backbone.body.0.0.weight',
            'roi_heads.box_predictor.cls_score.weight',
            'roi_heads.box_predictor.cls_score.bias',
            'roi_heads.box_predictor.bbox_pred.weight'
        ]
        
        for key in important_keys:
            if key in state_dict:
                shape = state_dict[key].shape
                print(f"   ✅ {key}: {shape}")
        
        # Phát hiện số classes
        if 'roi_heads.box_predictor.cls_score.weight' in state_dict:
            num_classes = state_dict['roi_heads.box_predictor.cls_score.weight'].shape[0]
            print(f"\n🎯 Số classes: {num_classes}")
        
        # Test load vào model
        print(f"\n🧪 TEST LOAD VÀO MODEL:")
        try:
            model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            
            # Tự động detect num_classes
            if 'roi_heads.box_predictor.cls_score.weight' in state_dict:
                detected_classes = state_dict['roi_heads.box_predictor.cls_score.weight'].shape[0]
                model.roi_heads.box_predictor = FastRCNNPredictor(in_features, detected_classes)
            
            model.load_state_dict(state_dict if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint else state_dict)
            model.to(device)
            model.eval()
            print(f"   ✅ Load thành công!")
            
            # Test inference
            dummy_input = torch.randn(1, 3, 320, 320).to(device)
            with torch.no_grad():
                output = model([dummy_input])
            print(f"   ✅ Inference test passed!")
            print(f"   📦 Output keys: {list(output[0].keys())}")
            
        except Exception as e:
            print(f"   ❌ Lỗi khi load: {e}")
        
        return checkpoint
        
    except Exception as e:
        print(f"❌ LỖI: Không thể đọc file - {e}")
        return None

# ==== SO SÁNH 2 MODELS ====
def compare_models(model1_path, model2_path):
    print("\n" + "="*60)
    print("🔄 SO SÁNH 2 MODELS")
    print("="*60)
    
    try:
        # Load 2 models
        ckpt1 = torch.load(model1_path, map_location=device, weights_only=False)
        ckpt2 = torch.load(model2_path, map_location=device, weights_only=False)
        
        # Extract state_dict
        if isinstance(ckpt1, dict) and 'model_state_dict' in ckpt1:
            state1 = ckpt1['model_state_dict']
        elif isinstance(ckpt1, dict) and 'model_state' in ckpt1:
            state1 = ckpt1['model_state']
        else:
            state1 = ckpt1
            
        if isinstance(ckpt2, dict) and 'model_state_dict' in ckpt2:
            state2 = ckpt2['model_state_dict']
        elif isinstance(ckpt2, dict) and 'model_state' in ckpt2:
            state2 = ckpt2['model_state']
        else:
            state2 = ckpt2
        
        # So sánh keys
        keys1 = set(state1.keys())
        keys2 = set(state2.keys())
        
        common_keys = keys1 & keys2
        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1
        
        print(f"\n📊 SO SÁNH KEYS:")
        print(f"   Model 1: {len(keys1)} layers")
        print(f"   Model 2: {len(keys2)} layers")
        print(f"   Chung: {len(common_keys)} layers")
        print(f"   Chỉ ở Model 1: {len(only_in_1)} layers")
        print(f"   Chỉ ở Model 2: {len(only_in_2)} layers")
        
        if only_in_1:
            print(f"\n   Layers chỉ có ở Model 1:")
            for key in list(only_in_1)[:5]:
                print(f"      - {key}")
        
        if only_in_2:
            print(f"\n   Layers chỉ có ở Model 2:")
            for key in list(only_in_2)[:5]:
                print(f"      - {key}")
        
        # So sánh weights của các layer chung
        print(f"\n🔍 SO SÁNH WEIGHTS (5 layers đầu):")
        for i, key in enumerate(list(common_keys)[:5]):
            w1 = state1[key]
            w2 = state2[key]
            
            if w1.shape == w2.shape:
                diff = (w1 - w2).abs().mean().item()
                max_diff = (w1 - w2).abs().max().item()
                print(f"   {key}:")
                print(f"      Shape: {w1.shape}")
                print(f"      Mean diff: {diff:.6f}")
                print(f"      Max diff: {max_diff:.6f}")
                print(f"      Giống nhau: {'✅ Có' if diff < 1e-6 else '❌ Không'}")
            else:
                print(f"   {key}: ⚠️ Shape khác nhau ({w1.shape} vs {w2.shape})")
        
        # So sánh metadata
        if isinstance(ckpt1, dict) and isinstance(ckpt2, dict):
            print(f"\n📋 SO SÁNH METADATA:")
            
            if 'best_map' in ckpt1 and 'best_map' in ckpt2:
                print(f"   Model 1 mAP: {ckpt1['best_map']:.4f}")
                print(f"   Model 2 mAP: {ckpt2['best_map']:.4f}")
                print(f"   Model tốt hơn: {'Model 1' if ckpt1['best_map'] > ckpt2['best_map'] else 'Model 2'}")
            
            if 'epoch' in ckpt1 and 'epoch' in ckpt2:
                print(f"   Model 1 epoch: {ckpt1['epoch']}")
                print(f"   Model 2 epoch: {ckpt2['epoch']}")
        
    except Exception as e:
        print(f"❌ LỖI khi so sánh: {e}")

# ==== MAIN ====
if __name__ == "__main__":
    print("🚀 BẮT ĐẦU PHÂN TÍCH MODELS")
    
    # Phân tích từng model
    checkpoint1 = load_and_analyze_model(model1_path, "Model 1 (best_model.pth)")
    checkpoint2 = load_and_analyze_model(model2_path, "Model 2 (img_best_model.pth)")
    
    # So sánh 2 models
    if checkpoint1 is not None and checkpoint2 is not None:
        compare_models(model1_path, model2_path)
    
    print("\n" + "="*60)
    print("✅ HOÀN THÀNH!")
    print("="*60)