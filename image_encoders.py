import torch
import torch.nn as nn
import torchvision
import numpy as np

from ddppo_resnet.resnet_policy import PNResnetDepthEncoder

class RGBEncoder(nn.Module):
    """RGB图像编码器类，基于ResNet50架构"""
    def __init__(self, resnet_pretrain=True, trainable=False):
        """
        初始化RGB编码器
        参数:
            resnet_pretrain: 是否使用预训练的ResNet50模型
            trainable: 是否允许编码器参数在训练中更新
        """
        super(RGBEncoder, self).__init__()
        if resnet_pretrain:
            print('\nLoading Torchvision pre-trained Resnet50 for RGB ...')
        # 加载ResNet50模型
        rgb_resnet = torchvision.models.resnet50(pretrained=resnet_pretrain)
        # 移除最后两层(全连接层和池化层)，只保留特征提取部分
        rgb_modules = list(rgb_resnet.children())[:-2]
        rgb_net = torch.nn.Sequential(*rgb_modules)
        self.rgb_net = rgb_net
        # 设置参数是否可训练
        for param in self.rgb_net.parameters():
            param.requires_grad_(trainable)

        # self.scale = 0.5  # 缩放因子(已注释)

    def forward(self, rgb_imgs):
        """
        前向传播函数
        参数:
            rgb_imgs: 形状为[B, N, C, H, W]的RGB图像张量，B为批次大小，N为每个样本的图像数
        返回:
            处理后的RGB特征
        """
        # 获取输入形状
        rgb_shape = rgb_imgs.size()
        # 重塑张量为[B*N, C, H, W]以便批量处理
        rgb_imgs = rgb_imgs.reshape(rgb_shape[0]*rgb_shape[1],
                                    rgb_shape[2], rgb_shape[3], rgb_shape[4])
        # 通过ResNet网络提取特征
        rgb_feats = self.rgb_net(rgb_imgs)  # * self.scale

        # 调试打印(已注释)
        # print('rgb_imgs', rgb_imgs.shape)
        # print('rgb_feats', rgb_feats.shape)

        # 返回压缩后的特征
        return rgb_feats.squeeze()


class DepthEncoder(nn.Module):
    """深度图像编码器类，基于PointNav的预训练ResNet"""
    def __init__(self, resnet_pretrain=True, trainable=False):
        """
        初始化深度编码器
        参数:
            resnet_pretrain: 是否使用预训练的ResNet模型
            trainable: 是否允许编码器参数在训练中更新
        """
        super(DepthEncoder, self).__init__()

        # 创建PointNav深度编码器实例
        self.depth_net = PNResnetDepthEncoder()
        if resnet_pretrain:
            print('Loading PointNav pre-trained Resnet50 for Depth ...')
            # 加载PointNav预训练权重
            ddppo_pn_depth_encoder_weights = torch.load('/home/vlnce/vln-ce/data/ddppo-models/gibson-2plus-resnet50.pth')
            # 提取视觉编码器相关的权重
            weights_dict = {}
            for k, v in ddppo_pn_depth_encoder_weights["state_dict"].items():
                split_layer_name = k.split(".")[2:]
                if split_layer_name[0] != "visual_encoder":
                    continue
                layer_name = ".".join(split_layer_name[1:])
                weights_dict[layer_name] = v
            # 释放原始权重变量以节省内存
            del ddppo_pn_depth_encoder_weights
            # 加载筛选后的权重到模型
            self.depth_net.load_state_dict(weights_dict, strict=True)
        # 设置参数是否可训练
        for param in self.depth_net.parameters():
            param.requires_grad_(trainable)

    def forward(self, depth_imgs):
        """
        前向传播函数
        参数:
            depth_imgs: 形状为[B, N, H, W, 1]的深度图像张量，B为批次大小，N为每个样本的图像数
        返回:
            处理后的深度特征
        """
        # 获取输入形状
        depth_shape = depth_imgs.size()
        # 重塑张量为[B*N, H, W, 1]以便批量处理
        depth_imgs = depth_imgs.reshape(depth_shape[0]*depth_shape[1],
                                    depth_shape[2], depth_shape[3], depth_shape[4])
        # 通过深度网络提取特征
        depth_feats = self.depth_net(depth_imgs)

        # 调试打印和断点(已注释)
        # print('depth_imgs', depth_imgs.shape)
        # print('depth_feats', depth_feats.shape)
        #
        # import pdb; pdb.set_trace()

        return depth_feats
