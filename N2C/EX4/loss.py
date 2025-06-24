#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEG信号去噪损失函数模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any


class MSELoss(nn.Module):
    """均方误差损失"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测信号 [batch_size, time, channels]
            target: 目标信号 [batch_size, time, channels]
        
        Returns:
            loss: MSE损失
        """
        loss = F.mse_loss(pred, target, reduction=self.reduction)
        return loss


class MAELoss(nn.Module):
    """平均绝对误差损失"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测信号 [batch_size, time, channels]
            target: 目标信号 [batch_size, time, channels]
        
        Returns:
            loss: MAE损失
        """
        loss = F.l1_loss(pred, target, reduction=self.reduction)
        return loss


class SNRLoss(nn.Module):
    """信噪比损失（负SNR作为损失）"""
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测信号 [batch_size, time, channels]
            target: 目标信号 [batch_size, time, channels]
        
        Returns:
            loss: 负SNR损失
        """
        # 计算信号功率
        signal_power = torch.mean(target ** 2, dim=(1, 2), keepdim=True)
        
        # 计算噪声功率（误差功率）
        noise_power = torch.mean((pred - target) ** 2, dim=(1, 2), keepdim=True)
        
        # 计算SNR (dB)
        snr_db = 10 * torch.log10(signal_power / (noise_power + self.eps) + self.eps)
        
        # 返回负SNR作为损失（最大化SNR = 最小化负SNR）
        return -torch.mean(snr_db)


class SpectralLoss(nn.Module):
    """频域损失"""
    
    def __init__(self, n_fft: int = 512, reduction: str = 'mean'):
        super().__init__()
        self.n_fft = n_fft
        self.reduction = reduction
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测信号 [batch_size, time, channels]
            target: 目标信号 [batch_size, time, channels]
        
        Returns:
            loss: 频域损失
        """
        batch_size, time_len, channels = pred.shape
        
        # 重塑为 [batch_size * channels, time]
        pred_flat = pred.view(-1, time_len)
        target_flat = target.view(-1, time_len)
        
        # 计算STFT
        pred_stft = torch.stft(
            pred_flat, n_fft=self.n_fft, return_complex=True, normalized=True
        )
        target_stft = torch.stft(
            target_flat, n_fft=self.n_fft, return_complex=True, normalized=True
        )
        
        # 计算幅度谱
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        
        # 计算频域MSE
        spectral_loss = F.mse_loss(pred_mag, target_mag, reduction=self.reduction)
        
        return spectral_loss


class PerceptualLoss(nn.Module):
    """感知损失（基于不同频段的加权）"""
    
    def __init__(self, freq_weights: Optional[torch.Tensor] = None):
        super().__init__()
        # 默认频段权重：低频更重要
        if freq_weights is None:
            freq_weights = torch.tensor([2.0, 1.5, 1.0, 0.8, 0.6])
        
        self.register_buffer('freq_weights', freq_weights)
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测信号 [batch_size, time, channels]
            target: 目标信号 [batch_size, time, channels]
        
        Returns:
            loss: 感知损失
        """
        # 简化版本：按不同频段计算加权MSE
        # 这里使用不同的滑动窗口来近似不同频段
        window_sizes = [50, 25, 10, 5, 2]  # 对应不同频率范围
        
        total_loss = 0.0
        
        for i, (window_size, weight) in enumerate(zip(window_sizes, self.freq_weights)):
            # 使用平均池化来模拟低通滤波
            if window_size > 1:
                # 对时间维度进行平均池化
                pred_filtered = F.avg_pool1d(
                    pred.transpose(1, 2), kernel_size=window_size, 
                    stride=1, padding=window_size//2
                ).transpose(1, 2)
                target_filtered = F.avg_pool1d(
                    target.transpose(1, 2), kernel_size=window_size,
                    stride=1, padding=window_size//2
                ).transpose(1, 2)
            else:
                pred_filtered = pred
                target_filtered = target
            
            # 计算该频段的MSE
            freq_loss = F.mse_loss(pred_filtered, target_filtered)
            total_loss += weight * freq_loss
        
        return total_loss / len(window_sizes)


class CombinedLoss(nn.Module):
    """组合损失函数"""
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        mae_weight: float = 0.0,
        snr_weight: float = 0.1,
        spectral_weight: float = 0.0,
        perceptual_weight: float = 0.0,
        **kwargs
    ):
        super().__init__()
        
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.snr_weight = snr_weight
        self.spectral_weight = spectral_weight
        self.perceptual_weight = perceptual_weight
        
        # 初始化各个损失函数
        self.mse_loss = MSELoss()
        
        if mae_weight > 0:
            self.mae_loss = MAELoss()
        
        if snr_weight > 0:
            self.snr_loss = SNRLoss()
        
        if spectral_weight > 0:
            self.spectral_loss = SpectralLoss(**kwargs.get('spectral_kwargs', {}))
        
        if perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss()
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: 预测信号 [batch_size, time, channels]
            target: 目标信号 [batch_size, time, channels]
        
        Returns:
            losses: 包含各个损失项的字典
        """
        losses = {}
        total_loss = 0.0
        
        # MSE损失
        if self.mse_weight > 0:
            mse_loss = self.mse_loss(pred, target)
            losses['mse'] = mse_loss
            total_loss += self.mse_weight * mse_loss
        
        # MAE损失
        if self.mae_weight > 0:
            mae_loss = self.mae_loss(pred, target)
            losses['mae'] = mae_loss
            total_loss += self.mae_weight * mae_loss
        
        # SNR损失
        if self.snr_weight > 0:
            snr_loss = self.snr_loss(pred, target)
            losses['snr'] = snr_loss
            total_loss += self.snr_weight * snr_loss
        
        # 频域损失
        if self.spectral_weight > 0:
            spectral_loss = self.spectral_loss(pred, target)
            losses['spectral'] = spectral_loss
            total_loss += self.spectral_weight * spectral_loss
        
        # 感知损失
        if self.perceptual_weight > 0:
            perceptual_loss = self.perceptual_loss(pred, target)
            losses['perceptual'] = perceptual_loss
            total_loss += self.perceptual_weight * perceptual_loss
        
        losses['total'] = total_loss
        
        return losses


def compute_snr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    计算信噪比（单独函数）
    
    Args:
        pred: 预测信号 [batch_size, time, channels]
        target: 目标信号 [batch_size, time, channels]
        eps: 防止除零的小值
    
    Returns:
        snr_db: SNR值（dB） [batch_size]
    """
    # 计算信号功率
    signal_power = torch.mean(target ** 2, dim=(1, 2))
    
    # 计算噪声功率（误差功率）
    noise_power = torch.mean((pred - target) ** 2, dim=(1, 2))
    
    # 计算SNR (dB)
    snr_db = 10 * torch.log10(signal_power / (noise_power + eps) + eps)
    
    return snr_db


def compute_pearson_correlation(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    计算皮尔逊相关系数
    
    Args:
        pred: 预测信号 [batch_size, time, channels]
        target: 目标信号 [batch_size, time, channels]
    
    Returns:
        correlation: 相关系数 [batch_size, channels]
    """
    batch_size, time_len, channels = pred.shape
    
    correlations = []
    
    for b in range(batch_size):
        batch_corr = []
        for c in range(channels):
            pred_c = pred[b, :, c]
            target_c = target[b, :, c]
            
            # 计算皮尔逊相关系数
            pred_mean = torch.mean(pred_c)
            target_mean = torch.mean(target_c)
            
            numerator = torch.sum((pred_c - pred_mean) * (target_c - target_mean))
            pred_std = torch.sqrt(torch.sum((pred_c - pred_mean) ** 2))
            target_std = torch.sqrt(torch.sum((target_c - target_mean) ** 2))
            
            correlation = numerator / (pred_std * target_std + 1e-8)
            batch_corr.append(correlation)
        
        correlations.append(torch.stack(batch_corr))
    
    return torch.stack(correlations)


def create_loss_function(config: Dict[str, Any]) -> nn.Module:
    """
    根据配置创建损失函数
    
    Args:
        config: 损失函数配置
    
    Returns:
        loss_fn: 损失函数实例
    """
    loss_type = config.get('type', 'mse')
    
    if loss_type == 'mse':
        return MSELoss()
    elif loss_type == 'mae':
        return MAELoss()
    elif loss_type == 'snr':
        return SNRLoss()
    elif loss_type == 'spectral':
        return SpectralLoss(**config.get('kwargs', {}))
    elif loss_type == 'perceptual':
        return PerceptualLoss()
    elif loss_type == 'combined':
        return CombinedLoss(**config.get('kwargs', {}))
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # 测试代码
    
    # 创建测试数据
    batch_size = 4
    time_len = 1000
    channels = 26
    
    target = torch.randn(batch_size, time_len, channels)
    pred = target + 0.1 * torch.randn_like(target)  # 添加噪声
    
    print("测试各种损失函数...")
    
    # 测试MSE损失
    mse_loss = MSELoss()
    mse_val = mse_loss(pred, target)
    print(f"MSE损失: {mse_val.item():.6f}")
    
    # 测试SNR损失
    snr_loss = SNRLoss()
    snr_val = snr_loss(pred, target)
    print(f"SNR损失: {snr_val.item():.6f}")
    
    # 测试组合损失
    combined_loss = CombinedLoss(
        mse_weight=1.0,
        snr_weight=0.1,
        perceptual_weight=0.05
    )
    
    combined_val = combined_loss(pred, target)
    print(f"组合损失:")
    for key, value in combined_val.items():
        print(f"  {key}: {value.item():.6f}")
    
    # 测试评估指标
    snr_db = compute_snr(pred, target)
    print(f"SNR (dB): {snr_db.mean().item():.2f} ± {snr_db.std().item():.2f}")
    
    correlation = compute_pearson_correlation(pred, target)
    print(f"相关系数: {correlation.mean().item():.3f} ± {correlation.std().item():.3f}") 