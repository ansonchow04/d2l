import torch
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt


# 下面的CenteredLayer类要从其输入中减去均值
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()  # 在前向传播中减去均值


# layer = CenteredLayer()
# print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))  # 测试自定义层

net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())

X = torch.rand(4, 8)  # 生成随机输入，分布均值接近0，标准差为1
Y = net(X)  # 测试包含自定义层的网络，生成随机输入
print(X, Y, Y.mean())  # 输出均值应接近0
# 使用散点图可视化输入和输出的分布
plt.scatter(range(X.numel()), X.flatten().detach().numpy(), label="input")
plt.legend()
plt.show()
