#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEG去噪模型
专门处理fT量级的MEG信号，使用成对的子采样信号进行训练
直接处理原始信号进行去噪
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalConv1D(nn.Module):
    """1D卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class ResidualBlock1D(nn.Module):
    """1D残差块"""
    def __init__(self, channels, kernel_size=3, dilation=1):
        super().__init__()
        
        self.conv1 = TemporalConv1D(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, 
                              padding=(kernel_size-1)*dilation//2, dilation=dilation)
        self.norm2 = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + residual
        return F.leaky_relu(out, 0.2)

class MEGDenoiser(nn.Module):
    """
    MEG去噪模型
    直接处理MEG信号，适合fT量级的数据
    """
    def __init__(self, n_channels=1, signal_length=1001):
        super().__init__()
        
        self.n_channels = n_channels
        self.signal_length = signal_length
        
        # 编码器 - 逐步提取特征
        self.encoder1 = TemporalConv1D(n_channels, 32, kernel_size=7, stride=1)
        self.encoder2 = TemporalConv1D(32, 64, kernel_size=5, stride=1)
        self.encoder3 = TemporalConv1D(64, 128, kernel_size=3, stride=1)
        
        # 残差块 - 深度特征提取
        self.res_blocks = nn.ModuleList([
            ResidualBlock1D(128, kernel_size=3, dilation=1),
            ResidualBlock1D(128, kernel_size=3, dilation=2),
            ResidualBlock1D(128, kernel_size=3, dilation=4),
            ResidualBlock1D(128, kernel_size=3, dilation=8),
            ResidualBlock1D(128, kernel_size=3, dilation=4),
            ResidualBlock1D(128, kernel_size=3, dilation=2),
            ResidualBlock1D(128, kernel_size=3, dilation=1),
        ])
        
        # 解码器 - 重建信号
        self.decoder1 = TemporalConv1D(128, 64, kernel_size=3, stride=1)
        self.decoder2 = TemporalConv1D(64, 32, kernel_size=5, stride=1)
        self.decoder3 = TemporalConv1D(32, 16, kernel_size=7, stride=1)
        
        # 输出层 - 直接输出去噪后的信号
        self.output = nn.Conv1d(16, n_channels, kernel_size=1)
        
        # 自适应层归一化 - 处理fT量级数据
        self.adaptive_norm = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        """
        前向传播
        x: [batch, channels, time] - MEG信号
        """
        # 保存输入用于残差连接
        input_signal = x
        
        # 编码
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        
        # 残差块处理
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # 解码
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        
        # 输出层
        output = self.output(x)
        
        # 确保输出与输入尺寸一致
        if output.shape[-1] != input_signal.shape[-1]:
            # 使用插值调整长度
            output = F.interpolate(output, size=input_signal.shape[-1], mode='linear', align_corners=False)
        
        # 残差连接 - 学习噪声而不是信号本身
        # output = input_signal - output  # 输出表示噪声，从输入中减去
        
        return output

class AttentionMEGDenoiser(nn.Module):
    """
    带注意力机制的MEG去噪模型
    """
    def __init__(self, n_channels=1, signal_length=1001):
        super().__init__()
        
        self.base_model = MEGDenoiser(n_channels, signal_length)
        
        # 时间注意力机制
        self.time_attention = nn.Sequential(
            nn.Conv1d(n_channels, n_channels//2 if n_channels > 1 else 1, 1),
            nn.ReLU(),
            nn.Conv1d(n_channels//2 if n_channels > 1 else 1, n_channels, 1),
            nn.Sigmoid()
        )
        
        # 全局特征
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_fc = nn.Sequential(
            nn.Linear(n_channels, n_channels*4),
            nn.ReLU(),
            nn.Linear(n_channels*4, n_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 基础去噪
        denoised = self.base_model(x)
        
        # 时间注意力
        time_att = self.time_attention(denoised)
        
        # 全局注意力
        global_feat = self.global_pool(denoised).squeeze(-1)  # [batch, channels]
        global_att = self.global_fc(global_feat).unsqueeze(-1)  # [batch, channels, 1]
        
        # 应用注意力
        output = denoised * time_att * global_att
        
        return output

class SimpleMEGDenoiser(nn.Module):
    """
    极简版MEG去噪模型
    确保项目能够跑通
    """
    def __init__(self, n_channels=1, signal_length=1001):
        super().__init__()
        
        self.n_channels = n_channels
        self.signal_length = signal_length
        
        # 极简网络架构
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        
        self.conv3 = nn.Conv1d(128, 64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.conv4 = nn.Conv1d(64, n_channels, kernel_size=7, padding=3)
        
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        # 保存输入
        input_signal = x
        
        # 前向传播
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        output = self.conv4(x)
        
        # 确保输出与输入尺寸一致
        if output.shape[-1] != input_signal.shape[-1]:
            output = F.interpolate(output, size=input_signal.shape[-1], mode='linear', align_corners=False)
        
        return output

# 为了保持兼容性，添加一个包装器
class MEGModelWrapper(nn.Module):
    """
    包装器类，保持与原有训练脚本的兼容性
    """
    def __init__(self, n_fft=None, hop_length=None, n_channels=1, model_type='simple'):
        super().__init__()
        
        # 这些参数不再使用，但保留以兼容原有接口
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_channels = n_channels
        
        # 选择模型类型
        if model_type == 'simple':
            self.model = SimpleMEGDenoiser(n_channels=n_channels)
        elif model_type == 'attention':
            self.model = AttentionMEGDenoiser(n_channels=n_channels)
        else:
            self.model = MEGDenoiser(n_channels=n_channels)
        
    def forward(self, x, n_fft=None, hop_length=None, is_istft=True):
        """
        兼容原有接口的前向传播
        x: 如果是STFT格式则忽略，如果是原始信号则直接处理
        """
        # 如果输入是STFT格式 [batch, channels, freq, time, 2]，则忽略
        # 这里我们假设会从数据加载器直接得到原始信号
        
        if len(x.shape) == 5:  # STFT格式，跳过
            print("警告：收到STFT格式输入，但模型无法处理。请修改数据加载器。")
            return torch.zeros(x.shape[0], x.shape[1], 500)
        
        # 处理原始信号 [batch, channels, time]
        output = self.model(x)
        
        return output 