import torch
from torch import nn

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
# print(net(X), X.shape, net(X).shape)  # 通过net进行前向计算

# print(net[2].state_dict())  # 访问第二个线性层的参数
# print(type(net[2].bias))
# print(net[2].bias)
# print(net[2].bias.data)
# print(net[2].bias.grad == None)

# 通过name_parameters访问和修改参数
# print(*[(name, param.shape) for name, param in net[0].named_parameters()])
# print(*[(name, param.shape) for name, param in net.named_parameters()])

# 通过state_dict访问和修改参数
# print(net.state_dict()["2.bias"].data)

def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f"block-{i}", block1())
    return net
    
# rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
# print(rgnet)
# print(rgnet[0][1][0].bias.data)  # 访问第二个子块的第一个线性层的偏置参数

def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
# net.apply(init_normal)
# print(net[0].weight.data[0], net[0].bias.data[0])  # 查看初始化结果

def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
# net.apply(init_constant)
# print(net[0].weight.data[0], net[0].bias.data[0])  # 查看初始化结果

def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

# net[0].apply(init_xavier)
# net[2].apply(init_42)
# print(net[0].weight.data)  # 查看初始化结果
# print(net[2].weight.data)  # 查看初始化结果

def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

# net.apply(my_init)
# print(net[0].weight[:2])

# net[0].weight.data[:] += 1
# net[0].weight.data[0, 0] = 42
# print(net[0].weight.data[0])

# 参数绑定，比如：在多个层间共享参数:我们可以定义一个稠密层,然后使用它的参数来设置另一个层的参数。
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), shared, nn.ReLU(), shared, nn.ReLU(), nn.Linear(8, 1))
print(net(X))

# print(net[2].weight.data[0] == net[4].weight.data[0])  # True
# net[2].weight.data[0, 0] = 100
# print(net[4].weight.data[0, 0])  # 100.0