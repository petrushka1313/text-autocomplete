# test_setup.py
import torch
import transformers

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Transformers version:", transformers.__version__)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))