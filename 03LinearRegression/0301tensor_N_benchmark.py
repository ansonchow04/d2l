#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 文件名称（建议）：tensor_add_benchmark.py
# 功能：矢量化加速演示与计时工具类 Timer；使用 PyTorch 对两个长度为 n 的张量逐元素相加并计时。
# 依赖：PyTorch、NumPy、time
# 用法：直接运行本脚本，最后打印耗时（单位：秒）。
# 创建日期：2025-10-22
# 备注：如需实际改名，请告知，我可以帮你在工作区重命名文件。


# 矢量化加速
import math
import time
import numpy as np
import torch

n = 10000
a = torch.ones([n])
b = torch.ones([n])

class Timer:
    """记录多次运行时间"""
    def __init__(self):
        self.times = [] # 列表，保存每次调用 stop() 时记录的时间（单位：秒）。初始为空列表。
        self.start()

    def start(self):
        """启动计时器，记录当前时间到 self.tik，用于开始或重置一次计时。"""
        self.tik = time.time() # 浮点数，记录上次调用 start() 时的时间戳（由 time.time() 返回）。
    
    def stop(self):
        """停止计时器并将时间记录在列表中，计算从 self.tik 到当前时间的时间差，将该差值追加到 self.times 列表中，并返回这次记录的时长"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]
    
    def avg(self):
        """返回平均时间，返回 self.times 列表中所有记录的平均值（sum / len）。"""
        return sum(self.times) / len(self.times)
    
    def sum(self):
        """返回时间总和，返回 self.times 列表中所有记录的和（即总耗时）。"""
        return sum(self.times)
    
    def cumsum(self):
        """返回累计时间，使用 NumPy 把 self.times 转为数组并调用 cumsum()，再把结果转换为 Python 列表，返回累计和序列。
    比如若 times = [0.1, 0.2, 0.15]，则 cumsum() 返回 [0.1, 0.3, 0.45]。"""
        return np.array(self.times).cumsum().tolist()

# 接下来对工作负载进行基准测试
# 使用for逐个相加
c = torch.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
print(f"{timer.stop():.5f} sec") #先执行 timer.stop() 得到一个浮点秒数；然后用:.5f保留5位小数；再拼接" sec"
# 使用重载的+运算符计算
timer.start()
d = a + b
print(f"{timer.stop():.5f} sec")