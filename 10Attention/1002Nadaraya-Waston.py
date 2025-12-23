import torch
from torch import nn
from d2l import torch as d2l
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

n_train = 50
x_train, _ = torch.sort(torch.rand(n_train) * 5)

def f(x):
    return 2 * torch.sin(x) + x**0.8

y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))
x_test = torch.arange(0, 5, 0.1)

y_truth = f(x_test)
n_test = len(x_test)
print(n_test)

def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'], xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)
    d2l.plt.show()

# 直接取平均值作为预测，平均汇聚，忽略了输入x的影响
y_hat = torch.repeat_interleave(y_train.mean(), n_test)
# plot_kernel_reg(y_hat)

# Nadaraya-Watson核回归，根据输入的位置进行加权
# 从注意力的角度重写之，得到更加通用的注意力汇聚公式：
# fx=∑α(x,xi)yi
# X_repeat.shape: (n_test*n_train),
# 每一行包含了相同的测试输入
# 构建Query矩阵：X_repeat; Key是x_train; 
X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
# -(X_repeat - x_train)**2 / 2用于计算查询与键之间的相似度
# softmax函数将相似度转换为权重attention_weights
# 以上就是注意力汇聚
attention_weights = nn.functional.softmax(-(X_repeat - x_train)**2 / 2, dim=1)
# value矩阵就是y_train
y_hat = torch.matmul(attention_weights, y_train)
# plot_kernel_reg(y_hat)
# d2l.show_heatmaps(attention_weights.reshape((1, 1, n_test, n_train)), xlabel='Sorted training inputs', ylabel='Sorted testing inputs')

# 使用小批量矩阵乘法，实现Nadaraya-Watson核回归
# 具体而言，首先将查询、键、值分别表示为三维张量，然后使用批量矩阵乘法计算加权和
class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))
    def forward(self, queries, keys, values):
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(-((queries - keys) ** 2) / (2 * self.w**2), dim=1)
        return torch.bmm(self.attention_weights.unsqueeze(1), values.unsqueeze(-1)).reshape(-1)
    
X_tile = x_train.repeat((n_train, 1))
Y_tile = y_train.repeat((n_train, 1))
keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

net = NWKernelRegression()
loss = nn.MSELoss(reduction='none')
trainer = torch.optim.SGD(net.parameters(), lr=0.05)
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])
# print(net(x_train, keys, values), y_train)

for epoch in range(5):
    trainer.zero_grad()
    l = loss(net(x_train, keys, values), y_train)
    l.sum().backward()
    trainer.step()
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
d2l.plt.show()
