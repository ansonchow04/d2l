#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 文件名称（建议）：0303_linear_regression_from_scratch.py
# 功能：线性回归的完整从零实现
#   1. 生成人工数据集（y = Xw + b + 噪声）
#   2. 实现小批量数据迭代器（data_iter）
#   3. 定义线性回归模型、均方损失函数、SGD 优化算法
#   4. 完整训练循环（前向传播、反向传播、参数更新）
# 依赖：PyTorch、D2L、Matplotlib
# 用法：直接运行脚本，输出每个 epoch 的损失并绘制数据分布图
# 说明：这是手动实现版本，未使用 PyTorch 的 nn.Module、DataLoader、optimizer 等高层 API

import random
import torch
from d2l import torch as d2l

# 构建一个人工数据集
# 服从均值为0的正态分布，标准差设置为0.01
def synthetic_data(w, b, num_examles):
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examles, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
#features:二维数据样本(输入x), labels:一维标签值(输出y)
features, labels = synthetic_data(true_w, true_b, 1000)

d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1)



def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples)) #range(x)生成0~x，list将range对象转换为列表
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

# 从均值为0，标准差为0.01的正态分布中采样随机数初始化权重，并将偏置初始化为0
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 定义模型，关联输入和输出，
def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b

# 定义损失函数
def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 定义优化算法
def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr *param.grad / batch_size
            param.grad.zero_()

lr = 0.03               # 学习率
num_epochs = 3          # 迭代周期个数
net = linreg            # 模型为线性回归模型
loss = squared_loss     # 损失函数为均方损失
batch_size = 10         # 批量大小为10
for epoch in range(num_epochs):                                         # 每一次迭代
    for X, y in data_iter(batch_size, features, labels):                # 通过data_iter随机获取10个样本的X和y
        l = loss(net(X, w, b), y)                                       # net返回了估计值y_hat，与y进行损失函数运算
        l.sum().backward()                                              # 对该批数据的损失求和，并使用backward()计算梯度
        sgd([w, b], lr, batch_size)                                     # 使用sgd算法更新w和b
    with torch.no_grad():                                               # 迭代周期结束后，禁用梯度计算 
        train_l = loss(net(features, w, b), labels)                     # 计算此次迭代结束后的损失并输出
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')