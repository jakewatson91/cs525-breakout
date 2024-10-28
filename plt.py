import torch

num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")