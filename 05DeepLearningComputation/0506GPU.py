import torch
from torch import nn

print(torch.device("cpu"), torch.device("cuda"), torch.device("cuda:1"))
print(torch.cuda.device_count())  # 可用GPU数量

# print(torch.version.cuda)
print(torch.backends.cudnn.enabled)
