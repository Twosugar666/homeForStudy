#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEG损失函数
专门用于fT量级的MEG信号训练
保持原有的自监督训练逻辑
"""

import torch
import torch.nn as nn
import numpy as np

class MEGLoss(nn.Module):
    """
    MEG损失函数
    保持原有的自监督训练逻辑
    """
    def __init__(self, gamma=1.0, alpha=1.0, beta=0.1):
        super().__init__()
        self.gamma = gamma  # 正则化权重
        self.alpha = alpha  # 主要损失权重
        self.beta = beta    # 一致性损失权重

    def mse_loss(self, pred, target, loss_mask=None):
        """
        均方误差损失，考虑fT量级数据的特点
        """
        loss = (pred - target) ** 2
        if loss_mask is not None:
            loss = loss * loss_mask
            return torch.sum(loss) / torch.sum(loss_mask + 1e-8)
        else:
            return torch.mean(loss)

    def l1_loss(self, pred, target, loss_mask=None):
        """
        L1损失，对fT量级数据更稳定
        """
        loss = torch.abs(pred - target)
        if loss_mask is not None:
            loss = loss * loss_mask
            return torch.sum(loss) / torch.sum(loss_mask + 1e-8)
        else:
            return torch.mean(loss)

    def temporal_consistency_loss(self, signal):
        """
        时间一致性损失 - 鼓励信号的平滑性
        适合MEG信号的时间连续性特点
        """
        # 计算时间差分
        diff1 = signal[:, :, 1:] - signal[:, :, :-1]  # 一阶差分
        diff2 = diff1[:, :, 1:] - diff1[:, :, :-1]    # 二阶差分
        
        # 平滑性损失
        smooth_loss = torch.mean(diff1 ** 2) + 0.1 * torch.mean(diff2 ** 2)
        return smooth_loss

    def wsdr_loss(self, g1_meg, fg1_meg, g2_meg, eps=1e-8):
        """
        加权源-失真比损失 (Weighted SDR)
        适配fT量级的MEG多通道数据
        """
        # 展平到 [batch, features]
        g1_flat = g1_meg.reshape(g1_meg.shape[0], -1)
        fg1_flat = fg1_meg.reshape(fg1_meg.shape[0], -1)
        g2_flat = g2_meg.reshape(g2_meg.shape[0], -1)
        
        def sdr_fn(true, pred, eps=1e-8):
            # 计算信噪比
            num = torch.sum(true * pred, dim=1)
            den = torch.norm(true, p=2, dim=1) * torch.norm(pred, p=2, dim=1)
            return -(num / (den + eps))

        # 计算噪声估计
        z_true = g1_flat - g2_flat  # 真实噪声
        z_pred = g1_flat - fg1_flat  # 预测噪声
        
        # 自适应权重，基于信号和噪声的相对强度
        signal_power = torch.sum(g2_flat ** 2, dim=1)
        noise_power = torch.sum(z_true ** 2, dim=1)
        a = signal_power / (signal_power + noise_power + eps)
        
        # 加权SDR
        wSDR = a * sdr_fn(g2_flat, fg1_flat) + (1 - a) * sdr_fn(z_true, z_pred)
        return torch.mean(wSDR)

    def regularization_loss(self, fg1_meg, g2_meg, g1fx, g2fx):
        """
        正则化损失 - 确保子采样一致性
        这是自监督学习的关键部分
        """
        return torch.mean((fg1_meg - g2_meg - g1fx + g2fx) ** 2)

    def signal_to_noise_ratio_loss(self, clean_signal, noisy_signal, denoised_signal):
        """
        信噪比损失 - 专门为MEG信号设计
        """
        # 估计噪声
        estimated_noise = noisy_signal - denoised_signal
        true_noise = noisy_signal - clean_signal
        
        # SNR损失
        signal_power = torch.mean(clean_signal ** 2, dim=-1, keepdim=True)
        noise_error = torch.mean((estimated_noise - true_noise) ** 2, dim=-1, keepdim=True)
        
        snr_loss = noise_error / (signal_power + 1e-8)
        return torch.mean(snr_loss)

    def correlation_loss(self, pred, target):
        """
        相关性损失 - 保持信号的相关结构
        """
        # 零均值化
        pred_centered = pred - torch.mean(pred, dim=-1, keepdim=True)
        target_centered = target - torch.mean(target, dim=-1, keepdim=True)
        
        # 计算相关系数
        numerator = torch.sum(pred_centered * target_centered, dim=-1)
        pred_norm = torch.sqrt(torch.sum(pred_centered ** 2, dim=-1) + 1e-8)
        target_norm = torch.sqrt(torch.sum(target_centered ** 2, dim=-1) + 1e-8)
        
        correlation = numerator / (pred_norm * target_norm + 1e-8)
        
        # 损失是1减去相关系数
        return torch.mean(1 - correlation)

    def forward(self, g1_meg, fg1_meg, g2_meg, g1fx, g2fx):
        """
        前向传播计算总损失
        
        参数:
        g1_meg: 子采样1的MEG数据 [batch, channels, time]
        fg1_meg: 网络对g1的预测输出 [batch, channels, time]
        g2_meg: 子采样2的MEG数据 [batch, channels, time]
        g1fx: 网络对完整噪声输入的子采样1 [batch, channels, time]
        g2fx: 网络对完整噪声输入的子采样2 [batch, channels, time]
        """
        
        # 1. 主要损失 - MSE和L1的组合
        loss_mse = self.mse_loss(fg1_meg, g2_meg)
        loss_l1 = self.l1_loss(fg1_meg, g2_meg)
        loss_main = 0.7 * loss_mse + 0.3 * loss_l1
        
        # 2. 时间一致性损失
        loss_temporal = self.temporal_consistency_loss(fg1_meg)
        
        # 3. 加权SDR损失 (自监督的核心)
        loss_wsdr = self.wsdr_loss(g1_meg, fg1_meg, g2_meg)
        
        # 4. 正则化损失 (保证子采样一致性)
        loss_reg = self.regularization_loss(fg1_meg, g2_meg, g1fx, g2fx)
        
        # 5. 相关性损失
        loss_corr = self.correlation_loss(fg1_meg, g2_meg)
        
        # 组合所有损失
        total_loss = (
            self.alpha * loss_main +           # 主要重建损失
            0.1 * loss_temporal +              # 时间平滑性
            loss_wsdr +                        # 信号质量
            self.gamma * loss_reg +            # 一致性约束
            self.beta * loss_corr              # 相关性保持
        )
        
        return total_loss

class SimpleMEGLoss(nn.Module):
    """
    简化版MEG损失函数
    确保项目能够跑通
    """
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, g1_meg, fg1_meg, g2_meg, g1fx, g2fx):
        """
        简化的损失计算
        """
        # 1. 基本MSE损失
        mse_loss = torch.mean((fg1_meg - g2_meg) ** 2)
        
        # 2. 简化的正则化损失
        reg_loss = torch.mean((fg1_meg - g2_meg - g1fx + g2fx) ** 2)
        
        # 3. L1损失增加稳定性
        l1_loss = torch.mean(torch.abs(fg1_meg - g2_meg))
        
        total_loss = mse_loss + 0.1 * l1_loss + self.gamma * reg_loss
        
        return total_loss

class AdaptiveMEGLoss(nn.Module):
    """
    自适应MEG损失函数
    根据信号特性动态调整损失权重
    """
    def __init__(self, base_loss=None):
        super().__init__()
        self.base_loss = base_loss or MEGLoss()
        
        # 可学习的权重参数
        self.main_weight = nn.Parameter(torch.tensor(1.0))
        self.consistency_weight = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, g1_meg, fg1_meg, g2_meg, g1fx, g2fx):
        # 计算基础损失
        base_loss = self.base_loss(g1_meg, fg1_meg, g2_meg, g1fx, g2fx)
        
        # 自适应权重调整
        main_component = torch.mean((fg1_meg - g2_meg) ** 2)
        consistency_component = torch.mean((fg1_meg - g2_meg - g1fx + g2fx) ** 2)
        
        # 应用可学习权重
        weighted_loss = (
            torch.sigmoid(self.main_weight) * main_component +
            torch.sigmoid(self.consistency_weight) * consistency_component
        )
        
        return base_loss + 0.1 * weighted_loss 

class SelfSupervisedMEGLoss(nn.Module):
    """
    纯自监督MEG损失函数
    Loss = gamma * ||f(g1(y)) - g2(y) - g1(f(y)) + g2(f(y))||^2
    其中：
    - g1(y), g2(y) 是从含噪数据中的采样子序列
    - f(g1(y)) 是模型对第一个子采样信号的输出
    - g1(f(y)), g2(f(y)) 是模型对含噪数据输出的两个子序列
    """
    def __init__(self, gamma=1.0):
        super(SelfSupervisedMEGLoss, self).__init__()
        self.gamma = gamma

    def forward(self, g1_y, fg1_y, g2_y, g1_fy, g2_fy):
        """
        参数:
        g1_y: 第一个子采样信号 (g1(y))
        fg1_y: 模型对第一个子采样信号的输出 (f(g1(y)))
        g2_y: 第二个子采样信号 (g2(y))
        g1_fy: 模型输出的第一个子采样 (g1(f(y)))
        g2_fy: 模型输出的第二个子采样 (g2(f(y)))
        """
        # 计算自监督损失
        reg_loss = torch.mean((fg1_y - g2_y - g1_fy + g2_fy) ** 2)
        
        # 应用gamma系数
        total_loss = self.gamma * reg_loss
        
        return total_loss 