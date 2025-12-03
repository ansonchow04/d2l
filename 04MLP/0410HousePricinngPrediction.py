import hashlib
import os
import tarfile
import zipfile
import requests

DATA_HUB = dict()  # 用于登记数据集名称与其对应的URL及校验码
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'  # 数据集下载基址


def download(name, cache_dir=os.path.join('..', 'data')):
    # 下载指定 name 的资源到本地缓存目录，并返回本地文件路径
    # name: 在 DATA_HUB 中登记的键；cache_dir: 缓存目录，默认 ../data
    assert name in DATA_HUB, f"{name}不存在于{DATA_HUB}"  # 确认 name 已登记，否则报错
    url, sha1_hash = DATA_HUB[name]  # 取出下载 URL 与期望的 SHA-1 校验值
    os.makedirs(cache_dir, exist_ok=True)  # 若缓存目录不存在则创建（已存在时不报错）
    fname = os.path.join(cache_dir, url.split('/')[-1])  # 目标文件路径=缓存目录/URL 最后段(文件名)
    if os.path.exists(fname):  # 如果本地已存在相同文件名，尝试做哈希校验复用
        sha1 = hashlib.sha1()  # 构造 SHA-1 计算器
        with open(fname, 'rb') as f:  # 以二进制只读打开，保证哈希用原始字节计算
            while True:  # 分块读取，避免大文件一次性读入内存
                data = f.read(1048576)  # 每次读取 1MiB (1<<20)
                if not data:  # 读到文件末尾
                    break
                sha1.update(data)  # 累计当前分块到哈希
        if sha1.hexdigest() == sha1_hash:  # 若哈希一致，直接复用已有文件
            return fname
    # 走到这里说明：文件不存在或哈希不符，需要重新下载
    print(f'正在从{url}下载{fname}...')  # 打印下载提示
    r = requests.get(url, stream=True, verify=True)  # 发起 HTTP 请求；stream=True 可用于流式下载
    with open(fname, 'wb') as f:  # 以二进制写模式打开目标文件（覆盖写入）
        f.write(r.content)  # 将响应内容写入文件（简洁写法；大文件可考虑分块写入）
    return fname  # 返回最终的本地文件路径


def download_extract(name, folder=None):
    # 下载并解压缩指定数据集，支持zip/tar格式
    fname = download(name)  # 调用上面的download函数
    base_dir = os.path.dirname(fname)  # 获取文件所在目录
    data_dir, ext = os.path.splitext(fname)  # 分离文件扩展名
    if ext == '.zip':  # 若为ZIP压缩包
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):  # 若为tar或gz压缩包
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'  # 若为其它格式则报错
    fp.extractall(base_dir)  # 解压至同级目录
    return os.path.join(base_dir, folder) if folder else data_dir  # 返回解压后的路径


def download_all():
    # 下载DATA_HUB中登记的所有数据集
    for name in DATA_HUB:
        download(name)


import numpy as np  # 科学计算库
import pandas as pd  # 数据处理库
import torch  # PyTorch 主库
from torch import nn  # 神经网络模块
from d2l import torch as d2l  # 李沐《动手学深度学习》工具包


# 登记Kaggle房价预测训练集及测试集的URL与SHA-1校验值
DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

# 下载数据集并读入Pandas DataFrame
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

print(train_data.shape, '\n', test_data.shape)  # 打印训练集和测试集的维度
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])  # 查看部分字段样例

# 拼接训练集与测试集（去掉标签列），统一做特征工程
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

# 取出数值型特征列的列名
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index

# 数值特征标准化（均值0，方差1）
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))

# 标准化后，若某列标准差为0或样本太少，可能产生NaN；将NaN填为0
all_features[numeric_features] = all_features[numeric_features].fillna(0.0)

# 对类别特征做独热编码；dummy_na=True表示为缺失值创建额外列
all_features = pd.get_dummies(all_features, dummy_na=True)

# 万一仍存在NaN（极少数全NaN列），统一填充为0
all_features = all_features.fillna(0.0)

print(all_features.shape)  # 打印独热编码后特征总维度

# 拆分回训练特征和测试特征
n_train = train_data.shape[0]  # 训练样本数
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)  # 训练特征张量
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)  # 测试特征张量
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)  # 标签张量

loss = nn.MSELoss()  # 均方误差损失函数
in_features = train_features.shape[1]  # 输入特征维度（线性层输入大小）


def get_net():
    # 定义线性模型 y = wX + b
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net


def log_rmse(net, features, labels):
    # 计算log均方根误差，用于房价预测任务
    clipped_preds = torch.clamp(net(features), 1, float('inf'))  # 防止对数取负
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))  # 计算log域下RMSE
    return rmse.item()  # 返回标量结果


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    # 模型训练函数，支持L2正则(weight_decay)
    train_ls, test_ls = [], []  # 用于记录训练和验证的loss曲线
    train_iter = d2l.load_array((train_features, train_labels), batch_size)  # 构建DataLoader
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)  # Adam优化器
    for epoch in range(num_epochs):  # 迭代多个epoch
        for X, y in train_iter:  # 批量读取数据
            optimizer.zero_grad()  # 清空梯度
            l = loss(net(X), y)  # 前向计算损失
            l.backward()  # 反向传播计算梯度
            optimizer.step()  # 参数更新
        train_ls.append(log_rmse(net, train_features, train_labels))  # 记录训练误差
        if test_labels is not None:  # 若存在验证集
            test_ls.append(log_rmse(net, test_features, test_labels))  # 记录验证误差
    return train_ls, test_ls  # 返回每个epoch的误差曲线


def get_k_fold_data(k, i, X, y):
    # 构造第i折的训练集和验证集（K折交叉验证）
    assert k > 1  # 至少两折
    fold_size = X.shape[0] // k  # 每折样本数
    X_train, y_train = None, None  # 初始化空集合
    for j in range(k):  # 遍历每一折
        idx = slice(j * fold_size, (j + 1) * fold_size)  # 当前折的索引范围
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part  # 第i折作为验证集
        elif X_train is None:
            X_train, y_train = X_part, y_part  # 初始化训练集
        else:
            X_train = torch.cat([X_train, X_part], 0)  # 其余折拼接为训练集
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid  # 返回划分后的数据集


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    # 执行K折交叉验证，返回平均训练误差和验证误差
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)  # 第i折数据
        net = get_net()  # 初始化模型
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)  # 训练一折
        train_l_sum += train_ls[-1]  # 累计最后一轮训练误差
        valid_l_sum += valid_ls[-1]  # 累计最后一轮验证误差
        if i == 0:
            # 绘制第1折的训练/验证误差曲线
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1},训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k  # 返回平均误差


# 设定超参数
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64


"""
# 打印数据基本信息
print('是否存在 SalePrice 缺失值:', train_data['SalePrice'].isnull().sum())
print('最小 SalePrice:', train_data['SalePrice'].min())
print('是否存在 NaN in train_labels:', torch.isnan(train_labels).any())

# 执行K折交叉验证
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)

# 打印平均训练/验证误差
print(f'{k}折验证平均训练log rmse：{float(train_l):f}, 平均验证log rmse: {float(valid_l):f}')
"
"""


def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch', ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse: {float(train_ls[-1]):f}')
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)

train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size)

# 显示误差曲线图
d2l.plt.show()