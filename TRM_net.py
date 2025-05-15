import torch
import torch.nn as nn
import numpy as np
import utils

from transformer.waypoint_bert import WaypointBert
from pytorch_transformers import BertConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def TRM_predict(mode, args, predictor, rgb_feats, depth_feats):
    ''' 预测路径点概率 '''
    # 使用预测器模型获取视觉逻辑值
    vis_logits = predictor(rgb_feats, depth_feats)
    # 逐元素概率 (使用sigmoid激活函数将逻辑值转换为0-1之间的概率)
    vis_probs = torch.sigmoid(vis_logits)

    # 根据模式返回不同的结果
    if mode == 'train':
        return vis_logits  # 训练模式返回逻辑值，用于计算损失
    elif mode == 'eval':
        return vis_probs, vis_logits  # 评估模式返回概率和逻辑值


class BinaryDistPredictor_TRM(nn.Module):
    """基于Transformer的二元分布预测器"""
    def __init__(self, args=None, hidden_dim=768, n_classes=12):
        """
        初始化预测器
        参数:
            args: 配置参数
            hidden_dim: 隐藏层维度，默认768
            n_classes: 输出类别数，默认12
        """
        super(BinaryDistPredictor_TRM, self).__init__()
        self.args = args
        self.batchsize = args.BATCH_SIZE
        self.num_angles = args.ANGLES  # 角度数
        self.num_imgs = args.NUM_IMGS  # 图像数
        self.n_classes = n_classes  # 输出类别数
        
        # RGB特征处理网络 (将ResNet特征映射到hidden_dim维度)
        # self.visual_1by1conv_rgb = nn.Conv2d(
        #     in_channels=2048, out_channels=512, kernel_size=1)
        self.visual_fc_rgb = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod([2048,7,7]), hidden_dim),  # 将2048x7x7的特征展平并投影到hidden_dim
            nn.ReLU(True),
        )
        
        # 深度特征处理网络 (将深度特征映射到hidden_dim维度)
        # self.visual_1by1conv_depth = nn.Conv2d(
        #     in_channels=128, out_channels=512, kernel_size=1)
        self.visual_fc_depth = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod([128,4,4]), hidden_dim),  # 将128x4x4的特征展平并投影到hidden_dim
            nn.ReLU(True),
        )
        
        # 融合RGB和深度特征的网络
        self.visual_merge = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),  # 将连接的特征降维回hidden_dim
            nn.ReLU(True),
        )

        # 配置Transformer模型
        config = BertConfig()
        config.model_type = 'visual'
        config.finetuning_task = 'waypoint_predictor'
        config.hidden_dropout_prob = 0.3
        config.hidden_size = 768
        config.num_attention_heads = 12
        config.num_hidden_layers = args.TRM_LAYER  # Transformer层数
        self.waypoint_TRM = WaypointBert(config=config)  # 初始化WaypointBert模型

        layer_norm_eps = config.layer_norm_eps
        # 层归一化(已注释)
        # self.mergefeats_LayerNorm = BertLayerNorm(
        #     hidden_dim,
        #     eps=layer_norm_eps
        # )

        # 创建注意力掩码，用于控制Transformer中的信息流动
        self.mask = utils.get_attention_mask(
            num_imgs=self.num_imgs,
            neighbor=args.TRM_NEIGHBOR).to(device)

        # 视觉分类器，将Transformer输出映射到最终预测
        self.vis_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,
                int(n_classes*(self.num_angles/self.num_imgs))),  # 输出维度适应角度和类别
        )

    def forward(self, rgb_feats, depth_feats):
        """
        前向传播函数
        参数:
            rgb_feats: RGB特征，来自RGB编码器
            depth_feats: 深度特征，来自深度编码器
        返回:
            vis_logits: 视觉逻辑值，形状为[batchsize, num_angles, n_classes]
        """
        # 计算每个批次中实际的样本数
        bsi = rgb_feats.size(0) // self.num_imgs

        # 处理RGB特征
        # rgb_x = self.visual_1by1conv_rgb(rgb_feats)
        rgb_x = self.visual_fc_rgb(rgb_feats).reshape(
            bsi, self.num_imgs, -1)  # 重塑为[bsi, num_imgs, hidden_dim]

        # 处理深度特征
        # depth_x = self.visual_1by1conv_depth(depth_feats)
        depth_x = self.visual_fc_depth(depth_feats).reshape(
            bsi, self.num_imgs, -1)  # 重塑为[bsi, num_imgs, hidden_dim]

        # 融合RGB和深度特征
        vis_x = self.visual_merge(
            torch.cat((rgb_x, depth_x), dim=-1)  # 沿最后一个维度连接
        )
        # 层归一化(已注释)
        # vis_x = self.mergefeats_LayerNorm(vis_x)

        # 为每个批次复制注意力掩码
        attention_mask = self.mask.repeat(bsi,1,1,1)
        # 通过Transformer处理融合特征
        vis_rel_x = self.waypoint_TRM(
            vis_x, attention_mask=attention_mask
        )

        # 应用视觉分类器得到逻辑值
        vis_logits = self.vis_classifier(vis_rel_x)
        # 重塑为[bsi, num_angles, n_classes]
        vis_logits = vis_logits.reshape(
            bsi, self.num_angles, self.n_classes)

        # 热图偏移（使每个图像指向中间）
        # 将后半部分和前半部分拼接，实现循环偏移
        vis_logits = torch.cat(
            (vis_logits[:,self.args.HEATMAP_OFFSET:,:], vis_logits[:,:self.args.HEATMAP_OFFSET,:]),
            dim=1)

        return vis_logits


class BertLayerNorm(nn.Module):
    """BERT风格的层归一化模块"""
    def __init__(self, hidden_size, eps=1e-12):
        """
        构造TensorFlow风格的层归一化模块（epsilon在平方根内部）
        参数:
            hidden_size: 隐藏层大小
            eps: 用于数值稳定的小常数
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 缩放参数
        self.bias = nn.Parameter(torch.zeros(hidden_size))  # 偏置参数
        self.variance_epsilon = eps  # 方差中添加的小常数，防止除零

    def forward(self, x):
        """
        前向传播
        参数:
            x: 输入张量
        返回:
            归一化后的张量
        """
        u = x.mean(-1, keepdim=True)  # 计算均值
        s = (x - u).pow(2).mean(-1, keepdim=True)  # 计算方差
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)  # 归一化
        return self.weight * x + self.bias  # 应用缩放和偏置
