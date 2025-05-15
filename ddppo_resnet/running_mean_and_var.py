#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor
from torch import distributed as distrib
from torch import nn as nn


class RunningMeanAndVar(nn.Module):
    """
    计算输入数据的运行均值和方差的模块
    用于实现类似BatchNorm的归一化，但保持运行统计量
    
    参数:
    n_channels (int): 输入特征的通道数
    """
    def __init__(self, n_channels: int) -> None:
        super().__init__()
        # 注册缓冲区，用于存储均值、方差和计数
        # 注册为缓冲区而非参数，这样它们会被保存但不会参与反向传播
        self.register_buffer("_mean", torch.zeros(1, n_channels, 1, 1))  # 初始化均值为0
        self.register_buffer("_var", torch.zeros(1, n_channels, 1, 1))   # 初始化方差为0
        self.register_buffer("_count", torch.zeros(()))                  # 初始化计数为0
        self._mean: torch.Tensor = self._mean
        self._var: torch.Tensor = self._var
        self._count: torch.Tensor = self._count

    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播函数：计算归一化后的输入
        
        参数:
        x (Tensor): 输入张量，形状为[B,C,H,W]
        
        返回:
        Tensor: 归一化后的张量，形状与输入相同
        """
        if self.training:
            # 在训练模式下，更新运行均值和方差
            n = x.size(0)  # 批次大小
            # 调整张量维度以加速通道维度上的计算
            # 将通道维度放到第一位，并将其余维度展平
            # 这样做可以提高数值稳定性，特别是在使用fp16精度时
            x_channels_first = (
                x.transpose(1, 0).contiguous().view(x.size(1), -1)
            )
            # 计算新的均值 (每个通道一个均值)
            new_mean = x_channels_first.mean(-1, keepdim=True)
            # 创建与_count形状相同的张量，填充当前批次大小
            new_count = torch.full_like(self._count, n)

            # 如果在分布式环境中，需要在所有进程间同步统计量
            if distrib.is_initialized():
                # 对新均值进行all-reduce操作（汇总所有进程的值）
                distrib.all_reduce(new_mean)
                distrib.all_reduce(new_count)
                # 除以进程数，得到全局均值
                new_mean /= distrib.get_world_size()

            # 计算新的方差 (每个通道一个方差)
            new_var = (
                (x_channels_first - new_mean).pow(2).mean(dim=-1, keepdim=True)
            )

            # 如果在分布式环境中，同步方差计算
            if distrib.is_initialized():
                distrib.all_reduce(new_var)
                new_var /= distrib.get_world_size()

            # 调整均值和方差的形状以匹配存储格式 [1,C,1,1]
            new_mean = new_mean.view(1, -1, 1, 1)
            new_var = new_var.view(1, -1, 1, 1)

            # 更新运行方差 - 使用Welford算法的变体
            # 该算法可以稳定地合并两个数据集的方差
            m_a = self._var * (self._count)
            m_b = new_var * (new_count)
            # 计算合并后的M2统计量（未归一化的方差）
            M2 = (
                m_a
                + m_b
                + (new_mean - self._mean).pow(2)
                * self._count
                * new_count
                / (self._count + new_count)
            )

            # 更新运行方差
            self._var = M2 / (self._count + new_count)
            # 更新运行均值 - 加权平均
            self._mean = (self._count * self._mean + new_count * new_mean) / (
                self._count + new_count
            )

            # 更新样本计数
            self._count += new_count

        # 计算归一化所需的逆标准差
        # 使用max函数确保方差不会太小，防止数值不稳定
        inv_stdev = torch.rsqrt(
            torch.max(self._var, torch.full_like(self._var, 1e-2))
        )
        # 计算归一化的输出: (x - mean) / std
        # 使用addcmul函数提高计算效率和数值稳定性
        # addcmul(value, tensor1, tensor2) 计算: value + tensor1 * tensor2
        return torch.addcmul(-self._mean * inv_stdev, x, inv_stdev)
