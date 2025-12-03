import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000) # 生成数据集

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays) # "*"为解包运算符，将data_arrays中的多个张量传递给TensorDataset构造函数，创建一个数据集对象
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

from torch import nn

net = nn.Sequential(nn.Linear(2, 1))
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss() # 平方L2泛数

trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs): # 每次迭代
    for X, y in data_iter:      # 访问一个批量的样本、特征、标签
        l = loss(net(X), y)
        trainer.zero_grad()     # 梯度清零
        l.backward()            # 计算梯度
        trainer.step()          # 读取grad，并根据优化算法（如 SGD）计算参数更新量，更形参数值
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
    weight_grad = net[0].weight.grad
    print(f"权重梯度形状: {weight_grad.shape}")  # torch.Size([1, 2])
    print(f"权重梯度值: {weight_grad}")

w = net[0].weight.data
b = net[0].bias.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
print('b的估计误差：', true_b - b)