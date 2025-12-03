import torch
from torch import nn

# functional：函数式接口
from torch.nn import functional as F

# 定义一个简单的神经网络,包含两个线性层和一个ReLU激活函数,256:隐藏单元,20:输入特征维度,10:输出类别数
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
print(X)

"""
在下面的代码片段中,我们从零开始编写一个块。
它包含一个多层感知机,其具有256个隐藏单元的隐藏层和一个10维输出层。
注意,下面的MLP类继承了表示块的类。我们的实现只需要提供我们自己的构造函数(Python中的__init__函数)和前向传播函数。
"""


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))


class mySequential(nn.Module):
    def __init__(self, *args):
        # 将每个模块依次添加到有序字典_modules中
        super().__init__()
        for idx, module in enumerate(args):
            # 使用add_module安全注册子模块（避免出现None或未注册的问题）
            self.add_module(str(idx), module)

    def forward(self, X):
        for block in self._modules.values():
            # 跳过为None的条目（防止“无法调用类型为'None'的对象”）
            if block is None:
                continue
            X = block(X)
        return X


# net = mySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
# print(net(X))


class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用不需要梯度的随机权重进行矩阵乘法，并添加偏置后应用ReLU激活函数，公式为：relu(XW + b)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        X = self.linear(X)
        # 控制流，直到X的绝对值和小于等于1，然后返回X的元素和
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


# net = FixedHiddenMLP()
# print(net(X))  # 前向传播


# 混合搭配各种组合块
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义一个包含多个层的序列网络,包括线性层和ReLU激活函数
        self.net = nn.Sequential(
            nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU()
        )
        # 定义一个线性层，将32维输入映射到16维输出
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        # 通过嵌套的序列网络处理输入X，然后通过线性层进行最终映射
        return self.linear(self.net(X))


chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
print(chimera(X))  # 前向传播
