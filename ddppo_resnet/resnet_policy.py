#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from habitat_baselines.rl.ddppo.policy import resnet


class PNResnetDepthEncoder(nn.Module):
    """
    基于ResNet的深度图像编码器
    用于将深度图像编码为固定维度的特征向量
    
    参数说明:
    baseplanes (int): 第一个卷积层的输出通道数，默认32
    ngroups (int): 组归一化(GroupNorm)的组数，默认16
    spatial_size (int): 输入图像的空间尺寸，默认128
    make_backbone: 主干网络构建函数，默认使用ResNet50
    """
    def __init__(
        self,
        baseplanes: int = 32,
        ngroups: int = 16,
        spatial_size: int = 128,
        make_backbone=getattr(resnet, 'resnet50'),
    ):
        super().__init__()

        # 设置输入深度图通道数为1
        self._n_input_depth = 1  # observation_space.spaces["depth"].shape[2]
        # 设置空间尺寸为原始尺寸的一半
        spatial_size = 256 // 2  # observation_space.spaces["depth"].shape[0]

        # 创建空的均值和方差运行统计模块
        self.running_mean_and_var = nn.Sequential()

        # 设置输入通道数
        input_channels = self._n_input_depth
        # 构建主干网络
        self.backbone = make_backbone(input_channels, baseplanes, ngroups)

        # 计算最终的空间尺寸
        final_spatial = int(
            spatial_size * self.backbone.final_spatial_compress # 压缩比例
        )
        # 设置压缩后的特征维度
        after_compression_flat_size = 2048
        # 计算压缩通道数
        num_compression_channels = int(
            round(after_compression_flat_size / (final_spatial ** 2))
        )
        
        # 构建压缩层
        self.compression = nn.Sequential(
            # 3x3卷积层，用于特征压缩
            nn.Conv2d(
                self.backbone.final_channels,
                num_compression_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            # 组归一化层
            nn.GroupNorm(1, num_compression_channels),
            # ReLU激活函数
            nn.ReLU(True),
        )

    def layer_init(self):
        """
        初始化网络层的参数
        使用Kaiming初始化方法初始化权重
        将偏置项初始化为0
        """
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, depth_observations):
        """
        前向传播函数
        
        参数:
        depth_observations: 输入的深度图像数据
        
        返回:
        x: 编码后的特征
        """
        cnn_input = []

        if self._n_input_depth > 0:
            # 调整张量维度顺序为 [批次 x 通道 x 高度 x 宽度]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            cnn_input.append(depth_observations)

        # 合并输入数据
        x = torch.cat(cnn_input, dim=1)
        # 使用平均池化进行下采样
        x = F.avg_pool2d(x, 2)

        # 通过均值和方差归一化
        x = self.running_mean_and_var(x)
        # 通过主干网络
        x = self.backbone(x)
        # 通过压缩层
        x = self.compression(x)
        return x
