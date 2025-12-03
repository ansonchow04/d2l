import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')  # 保存张量到文件
x2 = torch.load('x-file')  # 从文件加载张量
print(x2)

y = torch.zeros(4)
torch.save([x, y], 'xy-file')
x2, y2 = torch.load('xy-file')  # 从文件加载多个张量
print(x2, y2)

mydict = {'x': x, 'y': y }
torch.save(mydict, 'mydict-file')  # 保存字典到文件
mydict2 = torch.load('mydict-file')  # 从文件加载字典
print(mydict2)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
    
net = MLP()
X = torch.rand(size=(2, 20))
Y = net(X)
torch.save(net.state_dict(), 'mlp-params')  # 仅保存模型参数
clone = MLP()
clone.load_state_dict(torch.load('mlp-params'))  # 加载模型参数到新模型
print(clone.eval())
Y_clone = clone(X)
print(Y_clone == Y)  # 验证加载的模型参数是否正确