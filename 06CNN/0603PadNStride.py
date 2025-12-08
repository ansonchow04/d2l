import torch
from torch import nn

# 这里只是构建了输入输出接口，可以在调用时省略conv2d的标准输入输出格式的前两个维度（批量大小N和通道数C），没有对张量的形状产生任何影响
def comp_conv2d(conv2d, X):
    # 这里的(1,1)表示批量大小和通道数都是1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])

# 卷积核=3，填充=1，步长=1
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=1)
# X大小为8*8，经过填充会变成10*10
X = torch.rand(size=(8, 8))
# 输入10*10，卷积核3*3，步长1，遍历结果为(10-2)*(10-2)
print(comp_conv2d(conv2d, X).shape)
# 输入8*8，填充后变成8*10，卷积核3*5，步长3*4，遍历次数2*2
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
print(comp_conv2d(conv2d, X).shape)