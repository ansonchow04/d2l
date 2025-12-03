#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 文件名称：0302Normal_distribution_N_loss_of_squares.py
# 功能：计算并可视化不同均值与标准差（mu, sigma）的正态分布曲线。
# 依赖：NumPy、Matplotlib、D2L（PyTorch 版本工具包：from d2l import torch as d2l）
# 用法：直接运行本脚本，将弹出绘图窗口显示多条正态分布曲线。
# 说明：若环境中 d2l.plot 不可用，可改为直接使用 matplotlib（当前脚本已引入 plt.show()）。

import math
import numpy as np
from d2l import torch as d2l
import matplotlib.pyplot as plt

# 定义一个python函数来计算正态分布
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)

# 可视化正态分布
x = np.arange(-7, 7, 0.01)
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
plt.show()