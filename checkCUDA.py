import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")  # 第一块 GPU
    torch.cuda.set_device(0)         # 可选：显式设置 GPU
else:
    device = torch.device("cpu")

print("Using device:", device)
print(torch.__version__)       # 看torch版本
print(torch.version.cuda)      # 看PyTorch带的CUDA版本
print(torch.cuda.is_available())
