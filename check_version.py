import torch
import torchvision

print(f"PyTorch version: {torch.__version__}")
print(f"TorchVision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")

# Test NMS
try:
    boxes = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    scores = torch.tensor([0.9], dtype=torch.float32)
    keep = torchvision.ops.nms(boxes, scores, 0.5)
    print("✅ NMS works!")
except Exception as e:
    print(f"❌ NMS failed: {e}")