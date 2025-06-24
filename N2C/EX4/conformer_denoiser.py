#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conformer架构的MEG信号去噪模型
结合全局建模能力（注意力机制）和局部感知能力（卷积）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
import numpy as np


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query, key, value: [batch_size, seq_len, embed_dim]
            attn_mask: [seq_len, seq_len] or [batch_size, seq_len, seq_len]
        
        Returns:
            output: [batch_size, seq_len, embed_dim]
            attn_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = query.shape
        
        # 线性投影
        q = self.q_proj(query)  # [batch_size, seq_len, embed_dim]
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 重塑为多头形式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 形状: [batch_size, num_heads, seq_len, head_dim]
        
        # 计算注意力权重
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # 形状: [batch_size, num_heads, seq_len, seq_len]
        
        # 应用注意力掩码
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, v)
        # 形状: [batch_size, num_heads, seq_len, head_dim]
        
        # 重塑回原形状
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        return output, attn_weights


class DepthwiseConv1D(nn.Module):
    """深度可分离1D卷积"""
    
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        bias: bool = True
    ):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size, stride, padding, dilation,
            groups=channels, bias=bias
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.depthwise(x)


class ConvolutionModule(nn.Module):
    """Conformer中的卷积模块"""
    
    def __init__(
        self,
        embed_dim: int,
        kernel_size: int = 31,
        expansion_factor: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 点卷积扩展
        self.pointwise_conv1 = nn.Conv1d(embed_dim, embed_dim * expansion_factor, 1)
        
        # GLU激活
        self.glu = nn.GLU(dim=1)
        
        # 深度卷积
        self.depthwise_conv = DepthwiseConv1D(
            embed_dim, kernel_size, padding=(kernel_size - 1) // 2
        )
        
        # 批标准化
        self.batch_norm = nn.BatchNorm1d(embed_dim)
        
        # Swish激活
        self.swish = nn.SiLU()
        
        # 点卷积压缩
        self.pointwise_conv2 = nn.Conv1d(embed_dim, embed_dim, 1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
        
        Returns:
            output: [batch_size, seq_len, embed_dim]
        """
        # 转置为卷积格式 [batch_size, embed_dim, seq_len]
        x = x.transpose(1, 2)
        
        # 点卷积扩展
        x = self.pointwise_conv1(x)
        
        # GLU激活
        x = self.glu(x)
        
        # 深度卷积
        x = self.depthwise_conv(x)
        
        # 批标准化
        x = self.batch_norm(x)
        
        # Swish激活
        x = self.swish(x)
        
        # 点卷积压缩
        x = self.pointwise_conv2(x)
        
        # Dropout
        x = self.dropout(x)
        
        # 转置回序列格式
        x = x.transpose(1, 2)
        
        return x


class FeedForwardModule(nn.Module):
    """前馈网络模块"""
    
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        dropout: float = 0.1,
        activation: str = 'swish'
    ):
        super().__init__()
        
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
        
        Returns:
            output: [batch_size, seq_len, embed_dim]
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class ConformerBlock(nn.Module):
    """Conformer基本块"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: int,
        conv_kernel_size: int = 31,
        conv_expansion_factor: int = 2,
        ffn_expansion_factor: float = 4.0,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5
    ):
        super().__init__()
        
        # 第一个前馈网络 (scale = 0.5)
        self.ffn1 = FeedForwardModule(
            embed_dim, int(ffn_dim * ffn_expansion_factor), dropout
        )
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        
        # 多头注意力
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        
        # 卷积模块
        self.conv_module = ConvolutionModule(
            embed_dim, conv_kernel_size, conv_expansion_factor, dropout
        )
        self.norm3 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        
        # 第二个前馈网络 (scale = 0.5)
        self.ffn2 = FeedForwardModule(
            embed_dim, int(ffn_dim * ffn_expansion_factor), dropout
        )
        self.norm4 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
            attn_mask: attention mask
        
        Returns:
            output: [batch_size, seq_len, embed_dim]
        """
        # 第一个前馈网络 (scale = 0.5)
        residual = x
        x = self.norm1(x)
        x = self.ffn1(x)
        x = x * 0.5 + residual
        
        # 多头注意力
        residual = x
        x = self.norm2(x)
        x, _ = self.self_attn(x, x, x, attn_mask)
        x = self.dropout(x)
        x = x + residual
        
        # 卷积模块
        residual = x
        x = self.norm3(x)
        x = self.conv_module(x)
        x = x + residual
        
        # 第二个前馈网络 (scale = 0.5)
        residual = x
        x = self.norm4(x)
        x = self.ffn2(x)
        x = x * 0.5 + residual
        
        return x


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           -(math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
        
        Returns:
            output: [batch_size, seq_len, embed_dim]
        """
        return x + self.pe[:, :x.size(1)]


class ConformerEncoder(nn.Module):
    """Conformer编码器"""
    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        use_pos_encoding: bool = True
    ):
        super().__init__()
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        # 位置编码
        self.use_pos_encoding = use_pos_encoding
        if use_pos_encoding:
            self.pos_encoding = PositionalEncoding(embed_dim)
        
        # Conformer层
        self.layers = nn.ModuleList([
            ConformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # 输出层标准化
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
            attn_mask: attention mask
        
        Returns:
            output: [batch_size, seq_len, embed_dim]
        """
        # 输入投影
        x = self.input_projection(x)
        x = self.dropout(x)
        
        # 位置编码
        if self.use_pos_encoding:
            x = self.pos_encoding(x)
        
        # Conformer层
        for layer in self.layers:
            x = layer(x, attn_mask)
        
        # 输出层标准化
        x = self.layer_norm(x)
        
        return x


class ConformerDenoiser(nn.Module):
    """基于Conformer的MEG信号去噪模型"""
    
    def __init__(
        self,
        input_channels: int = 26,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        use_pos_encoding: bool = True,
        output_channels: Optional[int] = None
    ):
        """
        初始化Conformer去噪模型
        
        Args:
            input_channels: 输入MEG通道数
            embed_dim: 嵌入维度
            num_layers: Conformer层数
            num_heads: 注意力头数
            ffn_dim: 前馈网络维度
            conv_kernel_size: 卷积核大小
            dropout: Dropout比例
            use_pos_encoding: 是否使用位置编码
            output_channels: 输出通道数，默认与输入相同
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.embed_dim = embed_dim
        
        if output_channels is None:
            output_channels = input_channels
        self.output_channels = output_channels
        
        # Conformer编码器
        self.encoder = ConformerEncoder(
            input_dim=input_channels,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            conv_kernel_size=conv_kernel_size,
            dropout=dropout,
            use_pos_encoding=use_pos_encoding
        )
        
        # 输出投影层
        self.output_projection = nn.Linear(embed_dim, output_channels)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入噪声MEG信号 [batch_size, time, channels]
            attn_mask: 注意力掩码
        
        Returns:
            output: 去噪后的MEG信号 [batch_size, time, channels]
        """
        # 编码
        encoded = self.encoder(x, attn_mask)
        
        # 输出投影
        output = self.output_projection(encoded)
        
        return output
    
    def get_num_params(self) -> int:
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> dict:
        """获取模型信息"""
        return {
            'model_name': 'ConformerDenoiser',
            'input_channels': self.input_channels,
            'output_channels': self.output_channels,
            'embed_dim': self.embed_dim,
            'num_parameters': self.get_num_params(),
            'model_size_mb': self.get_num_params() * 4 / 1024 / 1024  # 假设float32
        }


def create_conformer_model(config: dict) -> ConformerDenoiser:
    """
    根据配置创建Conformer模型
    
    Args:
        config: 模型配置字典
    
    Returns:
        model: ConformerDenoiser模型实例
    """
    return ConformerDenoiser(
        input_channels=config.get('input_channels', 26),
        embed_dim=config.get('embed_dim', 256),
        num_layers=config.get('num_layers', 6),
        num_heads=config.get('num_heads', 8),
        ffn_dim=config.get('ffn_dim', 1024),
        conv_kernel_size=config.get('conv_kernel_size', 31),
        dropout=config.get('dropout', 0.1),
        use_pos_encoding=config.get('use_pos_encoding', True),
        output_channels=config.get('output_channels', None)
    )


if __name__ == "__main__":
    # 测试代码
    
    # 创建模型
    model_config = {
        'input_channels': 26,
        'embed_dim': 256,
        'num_layers': 4,
        'num_heads': 8,
        'ffn_dim': 1024,
        'conv_kernel_size': 31,
        'dropout': 0.1
    }
    
    model = create_conformer_model(model_config)
    
    # 打印模型信息
    print("模型信息:")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 测试前向传播
    batch_size = 4
    seq_len = 1000
    channels = 26
    
    # 创建随机输入
    x = torch.randn(batch_size, seq_len, channels)
    
    print(f"\n输入形状: {x.shape}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"输出形状: {output.shape}")
    print(f"输出数据范围: [{output.min():.3f}, {output.max():.3f}]")
    
    # 测试训练模式
    model.train()
    output_train = model(x)
    print(f"训练模式输出形状: {output_train.shape}") 