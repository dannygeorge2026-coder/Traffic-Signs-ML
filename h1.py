import torch
print(torch.cuda.is_available())   # should be True
print(torch.version.cuda)          # should show 13.0
print(torch.cuda.get_device_name(0))  # should show "NVIDIA GeForce RTX 3050"
