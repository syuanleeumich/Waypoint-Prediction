# Copyright (c) 2020 Microsoft Corporation. Licensed under the MIT license.
# Modified in Recurrent VLN-BERT, 2020, Yicong.Hong@anu.edu.au

from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from .pytorch_transformer.modeling_bert import (BertEmbeddings,
        BertSelfAttention, BertAttention, BertEncoder, BertLayer,
        BertSelfOutput, BertIntermediate, BertOutput,
        BertPooler, BertLayerNorm, BertPreTrainedModel,
		BertPredictionHeadTransform)

logger = logging.getLogger(__name__)

class VisPosEmbeddings(nn.Module):
    """
    视觉位置嵌入模块
    用于为视觉特征添加位置信息，类似于BERT中的位置编码
    
    参数:
    config: 配置对象，包含hidden_size等参数
    """
    def __init__(self, config):
        super(VisPosEmbeddings, self).__init__()
        # 位置嵌入层，支持最多24个位置，每个位置用hidden_size维的向量表示
        self.position_embeddings = nn.Embedding(24, config.hidden_size)
        # 层归一化，用于稳定训练
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_vis_feats, position_ids=None):
        """
        前向传播函数
        
        参数:
        input_vis_feats: 输入的视觉特征，形状 [batch_size, seq_length, hidden_size]
        position_ids: 位置ID，如果为None则自动生成
        
        返回:
        embeddings: 添加位置信息后的特征，形状 [batch_size, seq_length, hidden_size]
        """
        seq_length = input_vis_feats.size(1)
        if position_ids is None:
            # 如果没有提供位置ID，则自动生成从0到seq_length-1的位置ID
            # 形状: [seq_length]
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_vis_feats.device)
            # shape: [batch_size, seq_length] - 将position_ids扩展到batch维度
            position_ids = position_ids.unsqueeze(0).repeat(input_vis_feats.size(0), 1)

        # 形状: [batch_size, seq_length, hidden_size]
        vis_embeddings = input_vis_feats
        # 获取位置嵌入，形状: [batch_size, seq_length, hidden_size]
        position_embeddings = self.position_embeddings(position_ids)

        # 将视觉特征与位置嵌入相加，形状: [batch_size, seq_length, hidden_size]
        embeddings = vis_embeddings + position_embeddings
        # 应用层归一化，形状不变: [batch_size, seq_length, hidden_size]
        embeddings = self.LayerNorm(embeddings)
        # embeddings = self.dropout(embeddings)
        return embeddings

class CaptionBertSelfAttention(BertSelfAttention):
    """
    修改自BertSelfAttention，添加了对history_state的支持
    用于处理带有历史信息的自注意力机制
    """
    def __init__(self, config):
        super(CaptionBertSelfAttention, self).__init__(config)
        self.config = config

    def forward(self, hidden_states, attention_mask, head_mask=None,
            history_state=None):
        """
        前向传播函数
        
        参数:
        hidden_states: 当前隐藏状态，形状 [batch_size, seq_length, hidden_size]
        attention_mask: 注意力掩码，形状 [batch_size, 1, 1, seq_length]或[batch_size, 1, 1, seq_length+history_length]
        head_mask: 注意力头掩码，形状 [num_heads] 或 None
        history_state: 历史状态，用于跨时间步的注意力计算，形状 [batch_size, history_length, hidden_size] 或 None
        
        返回:
        outputs: 包含上下文层和注意力分数的元组
          - context_layer: 形状 [batch_size, seq_length, hidden_size]
          - attention_scores: 形状 [batch_size, num_heads, seq_length, seq_length]或[batch_size, num_heads, seq_length, seq_length+history_length]
        """
        if history_state is not None:
            # 如果有历史状态，将其与当前状态连接作为key和value
            # 形状: [batch_size, history_length + seq_length, hidden_size]
            x_states = torch.cat([history_state, hidden_states], dim=1)
            # 查询仍然只使用当前状态，形状: [batch_size, seq_length, hidden_size]
            mixed_query_layer = self.query(hidden_states)  
            # 键使用历史+当前状态，形状: [batch_size, history_length + seq_length, hidden_size]
            mixed_key_layer = self.key(x_states)           
            # 值使用历史+当前状态，形状: [batch_size, history_length + seq_length, hidden_size]
            mixed_value_layer = self.value(x_states)       
        else:
            # 如果没有历史状态，使用标准的自注意力
            # 所有输出形状都是 [batch_size, seq_length, hidden_size]
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        ''' language feature only provide Keys and Values '''
        # 调整维度以便进行多头注意力计算
        # 形状: [batch_size, num_attention_heads, seq_length, attention_head_size]
        # 其中 hidden_size = num_attention_heads * attention_head_size
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 计算注意力分数: Q * K^T
        # 形状: [batch_size, num_attention_heads, seq_length, seq_length]或[batch_size, num_attention_heads, seq_length, seq_length+history_length]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # 缩放注意力分数
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # 应用注意力掩码
        attention_scores = attention_scores + attention_mask

        # 将注意力分数归一化为概率
        # 形状: [batch_size, num_attention_heads, seq_length, seq_length]或[batch_size, num_attention_heads, seq_length, seq_length+history_length]
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # 应用dropout，形状不变
        attention_probs = self.dropout(attention_probs)

        # 应用头部掩码（如果有）
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文层: attention_probs * V
        # 形状: [batch_size, num_attention_heads, seq_length, attention_head_size]
        context_layer = torch.matmul(attention_probs, value_layer)

        # 调整维度
        # 形状: [batch_size, seq_length, num_attention_heads, attention_head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # 形状: [batch_size, seq_length, hidden_size]
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_scores)

        return outputs


class CaptionBertAttention(BertAttention):
    """
    修改自BertAttention，集成了修改后的CaptionBertSelfAttention
    """
    def __init__(self, config):
        super(CaptionBertAttention, self).__init__(config)
        self.self = CaptionBertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.config = config

    def forward(self, input_tensor, attention_mask, head_mask=None,
            history_state=None):
        """
        前向传播函数
        
        参数:
        input_tensor: 输入张量，形状 [batch_size, seq_length, hidden_size]
        attention_mask: 注意力掩码，形状 [batch_size, 1, 1, seq_length]或[batch_size, 1, 1, seq_length+history_length]
        head_mask: 头部掩码，形状 [num_heads] 或 None
        history_state: 历史状态，形状 [batch_size, history_length, hidden_size] 或 None
        
        返回:
        outputs: 包含注意力输出和注意力分数的元组
          - attention_output: 形状 [batch_size, seq_length, hidden_size]
          - attention_scores: 形状 [batch_size, num_heads, seq_length, seq_length]或[batch_size, num_heads, seq_length, seq_length+history_length]
        """
        ''' transformer processing '''
        # 自注意力处理
        # self_outputs[0]: context_layer，形状 [batch_size, seq_length, hidden_size]
        # self_outputs[1]: attention_scores，形状 [batch_size, num_heads, seq_length, seq_length]或[batch_size, num_heads, seq_length, seq_length+history_length]
        self_outputs = self.self(input_tensor, attention_mask, head_mask, history_state)

        ''' feed-forward network with residule '''
        # 前馈网络处理，带有残差连接
        # 形状: [batch_size, seq_length, hidden_size]
        attention_output = self.output(self_outputs[0], input_tensor)

        # 组合输出
        outputs = (attention_output,) + self_outputs[1:]  # 如果输出注意力分数，则添加

        return outputs


class CaptionBertLayer(BertLayer):
    """
    修改自BertLayer，使用修改后的CaptionBertAttention
    """
    def __init__(self, config):
        super(CaptionBertLayer, self).__init__(config)
        self.attention = CaptionBertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None,
                history_state=None):
        """
        前向传播函数
        
        参数:
        hidden_states: 隐藏状态，形状 [batch_size, seq_length, hidden_size]
        attention_mask: 注意力掩码，形状 [batch_size, 1, 1, seq_length]或[batch_size, 1, 1, seq_length+history_length]
        head_mask: 头部掩码，形状 [num_heads] 或 None
        history_state: 历史状态，形状 [batch_size, history_length, hidden_size] 或 None
        
        返回:
        outputs: 层输出和注意力分数
          - layer_output: 形状 [batch_size, seq_length, hidden_size]
          - attention_scores: 形状 [batch_size, num_heads, seq_length, seq_length]或[batch_size, num_heads, seq_length, seq_length+history_length]
        """
        # 注意力层处理
        # attention_outputs[0]: attention_output，形状 [batch_size, seq_length, hidden_size]
        # attention_outputs[1]: attention_scores，形状 [batch_size, num_heads, seq_length, seq_length]或[batch_size, num_heads, seq_length, seq_length+history_length]
        attention_outputs = self.attention(hidden_states, attention_mask,
                head_mask, history_state)

        ''' feed-forward network with residule '''
        # 前馈网络处理，带有残差连接
        attention_output = attention_outputs[0]
        # 中间层（通常是扩展维度的线性层+激活函数）
        # 形状: [batch_size, seq_length, intermediate_size]，其中intermediate_size通常是hidden_size的4倍
        intermediate_output = self.intermediate(attention_output)
        # 输出层（通常是压缩维度的线性层+残差连接）
        # 形状: [batch_size, seq_length, hidden_size]
        layer_output = self.output(intermediate_output, attention_output)
        
        # 组合输出
        outputs = (layer_output,) + attention_outputs[1:]

        return outputs


class CaptionBertEncoder(BertEncoder):
    """
    修改自BertEncoder，使用修改后的CaptionBertLayer
    """
    def __init__(self, config):
        super(CaptionBertEncoder, self).__init__(config)
        self.output_attentions = config.output_attentions 
        self.output_hidden_states = config.output_hidden_states
        # 12 Bert layers
        # 创建多个Transformer层
        self.layer = nn.ModuleList([CaptionBertLayer(config) for _ in range(config.num_hidden_layers)])
        self.config = config

    def forward(self, hidden_states, attention_mask, head_mask=None,
                encoder_history_states=None):
        """
        前向传播函数
        
        参数:
        hidden_states: 隐藏状态，形状 [batch_size, seq_length, hidden_size]
        attention_mask: 注意力掩码，形状 [batch_size, 1, 1, seq_length]
        head_mask: 头部掩码，形状 [num_layers, num_heads] 或 None
        encoder_history_states: 编码器历史状态，形状 [num_layers, batch_size, history_length, hidden_size] 或 None
        
        返回:
        outputs: 包含隐藏状态和注意力分数的元组
          - hidden_states: 形状 [batch_size, seq_length, hidden_size]
          - slang_attention_score: 形状 [batch_size, num_heads, seq_length, seq_length]或[batch_size, num_heads, seq_length, seq_length+history_length]
        """
        # 遍历所有层
        for i, layer_module in enumerate(self.layer):
            # 获取当前层的历史状态（如果有）
            # 如果有历史状态，形状为 [batch_size, history_length, hidden_size]
            history_state = None if encoder_history_states is None else encoder_history_states[i] # default None

            # 通过当前层处理
            # layer_outputs[0]: layer_output，形状 [batch_size, seq_length, hidden_size]
            # layer_outputs[1]: attention_scores，形状 [batch_size, num_heads, seq_length, seq_length]或[batch_size, num_heads, seq_length, seq_length+history_length]
            layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i],
                    history_state)
            # 更新隐藏状态
            hidden_states = layer_outputs[0]

            # 保存最后一层的注意力分数
            if i == self.config.num_hidden_layers - 1:
                slang_attention_score = layer_outputs[1]

        # 组合输出
        outputs = (hidden_states, slang_attention_score)

        return outputs


class BertImgModel(nn.Module):
    """
    扩展自BertModel，用于处理图像区域特征作为输入
    """
    def __init__(self, config):
        super(BertImgModel, self).__init__()
        self.config = config
        # self.vis_pos_embeds = VisPosEmbeddings(config)
        # 创建编码器
        self.encoder = CaptionBertEncoder(config)

    def forward(self, input_x, attention_mask=None):
        """
        前向传播函数
        
        参数:
        input_x: 输入特征，形状 [batch_size, seq_length, hidden_size]
        attention_mask: 注意力掩码，形状 [batch_size, seq_length]
        
        返回:
        outputs: 最终编码后的输出
          - encoder_outputs[0]: 隐藏状态，形状 [batch_size, seq_length, hidden_size]
          - encoder_outputs[1]: 注意力分数，形状 [batch_size, num_heads, seq_length, seq_length]
        """
        # 调整注意力掩码
        # 转换为与模型参数相同的数据类型
        extended_attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        # 将0->-10000.0(需要mask的位置)，1->0.0(不需要mask的位置)
        # 形状: [batch_size, 1, 1, seq_length]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # 设置头部掩码
        head_mask = [None] * self.config.num_hidden_layers

        ''' positional encodings '''
        # 位置编码（当前被注释）
        # input_x = self.vis_pos_embeds(input_x)

        ''' pass to the Transformer layers '''
        # 通过Transformer层
        encoder_outputs = self.encoder(input_x,
                extended_attention_mask, head_mask=head_mask)

        # 组合输出
        outputs = (encoder_outputs[0],) + encoder_outputs[1:]

        return outputs


class WaypointBert(nn.Module):
    """
    修改自BertForMultipleChoice，支持路径点预测任务
    """
    def __init__(self, config=None):
        super(WaypointBert, self).__init__()
        self.config = config
        # BERT图像模型
        self.bert = BertImgModel(config)
        # Dropout层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_x, attention_mask=None):
        """
        前向传播函数
        
        参数:
        input_x: 输入特征，形状 [batch_size, seq_length, hidden_size]
        attention_mask: 注意力掩码，形状 [batch_size, seq_length]
        
        返回:
        sequence_output: 序列输出，形状 [batch_size, seq_length, hidden_size]
        """
        # 通过BERT模型
        outputs = self.bert(input_x, attention_mask=attention_mask)

        # 获取序列输出并应用dropout
        # 形状: [batch_size, seq_length, hidden_size]
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        return sequence_output
