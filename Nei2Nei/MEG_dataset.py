#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEG数据加载器
专门处理fT量级的MEG信号
直接使用成对的子采样信号进行训练
"""

import torch
import numpy as np
import scipy.io
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path

def subsample2_meg(meg_data):  
    """
    MEG数据的子采样函数
    meg_data: shape [channels, time_points]
    返回两个子采样信号
    """
    k = 2
    if torch.is_tensor(meg_data):
        meg_data = meg_data.cpu().numpy()
    
    channels, dim = meg_data.shape 
    dim1 = dim // k - 64  # 调整MEG数据的修正参数
    
    if dim1 <= 0:
        dim1 = dim // k  # 如果太小，直接使用一半
    
    meg1, meg2 = np.zeros([channels, dim1]), np.zeros([channels, dim1])
    
    for channel in range(channels):
        for i in range(dim1):
            i1 = i * k
            if i1 + 1 < dim:  # 确保不越界
                num = np.random.choice([0, 1])
                if num == 0:
                    meg1[channel, i], meg2[channel, i] = meg_data[channel, i1], meg_data[channel, i1+1]
                elif num == 1:
                    meg1[channel, i], meg2[channel, i] = meg_data[channel, i1+1], meg_data[channel, i1]
            else:
                meg1[channel, i], meg2[channel, i] = meg_data[channel, i1], meg_data[channel, i1]

    return torch.from_numpy(meg1).float(), torch.from_numpy(meg2).float()

class MEGDataset(Dataset):
    """
    MEG数据集
    直接处理原始信号
    """
    def __init__(self, noisy_files, clean_files, max_length=1001):
        super().__init__()
        
        self.noisy_files = sorted(noisy_files)
        self.clean_files = sorted(clean_files)
        self.max_length = max_length
        self.len_ = len(self.noisy_files)
        self.n_channels = 1  # MEG通道数

    def __len__(self):
        return self.len_

    def load_meg_sample(self, file_path):
        """
        加载MEG .mat文件，返回信号
        返回: torch.Tensor, shape [n_channels, time_points]
        """
        try:
            mat_data = scipy.io.loadmat(str(file_path))  # 确保使用字符串路径
            
            # 寻找数据键
            if 'data' in mat_data:
                meg_data = mat_data['data']
            elif 'B1' in mat_data:
                meg_data = mat_data['B1']
            else:
                # 尝试找到第一个非元数据的键
                data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                if data_keys:
                    meg_data = mat_data[data_keys[0]]
                else:
                    raise ValueError("无法找到数据键")
            
            # 确保数据形状为 [channels, time_points]
            if meg_data.shape[0] > meg_data.shape[1]:
                meg_data = meg_data.T
                
            # 选择指定数量的通道
            if meg_data.shape[0] > self.n_channels:
                meg_data = meg_data[:self.n_channels, :]
            
            # 处理fT量级数据的数值稳定性
            # 将数据缩放到合适的范围，避免数值下溢
            if np.max(np.abs(meg_data)) > 0:
                # 保持相对大小关系，但缩放到更大的数值范围
                scale_factor = 1e15  # 将fT缩放到pT量级
                meg_data = meg_data * scale_factor
            
            return torch.from_numpy(meg_data.astype(np.float32))
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # 返回零数据作为fallback
            return torch.zeros(self.n_channels, self.max_length, dtype=torch.float32)

    def _prepare_meg_sample(self, meg_data):
        """
        准备MEG样本：填充或截断到固定长度
        """
        channels, current_len = meg_data.shape
        
        output = torch.zeros((channels, self.max_length), dtype=torch.float32)
        
        if current_len >= self.max_length:
            # 截断
            output = meg_data[:, :self.max_length]
        else:
            # 填充
            output[:, :current_len] = meg_data
            
        return output

    def __getitem__(self, index):
        # 加载clean和noisy MEG数据
        x_clean = self.load_meg_sample(self.clean_files[index])
        x_noisy = self.load_meg_sample(self.noisy_files[index])
        
        # 填充/截断到固定长度
        x_clean = self._prepare_meg_sample(x_clean)
        x_noisy = self._prepare_meg_sample(x_noisy)
        
        # 子采样生成训练对
        g1_meg, g2_meg = subsample2_meg(x_noisy)
        
        # 确保数据类型
        x_clean = x_clean.type(torch.FloatTensor)
        x_noisy = x_noisy.type(torch.FloatTensor)
        g1_meg = g1_meg.type(torch.FloatTensor)
        g2_meg = g2_meg.type(torch.FloatTensor)
        
        # 返回原始信号
        # 为了保持与原有训练脚本的兼容性，我们返回相同的结构
        return x_noisy, g1_meg, x_clean, g2_meg, x_noisy  # 最后一个参数是为了兼容性

class MEGBatchDataset(Dataset):
    """
    批处理优化的MEG数据集
    """
    def __init__(self, noisy_files, clean_files, max_length=1001, use_augmentation=False):
        super().__init__()
        
        self.noisy_files = sorted(noisy_files)
        self.clean_files = sorted(clean_files)
        self.max_length = max_length
        self.len_ = len(self.noisy_files)
        self.n_channels = 1
        self.use_augmentation = use_augmentation
        
    def __len__(self):
        return self.len_
    
    def _augment_signal(self, signal):
        """
        信号增强：添加小量噪声，时间平移等
        """
        if not self.use_augmentation:
            return signal
        
        # 添加小量高斯噪声 (保持fT量级)
        noise_level = torch.std(signal) * 0.01  # 1%的噪声
        noise = torch.randn_like(signal) * noise_level
        
        # 时间平移
        shift = np.random.randint(-10, 11)  # 最多平移10个时间点
        if shift != 0:
            signal = torch.roll(signal, shift, dims=-1)
        
        return signal + noise
    
    def load_meg_sample(self, file_path):
        """加载MEG样本"""
        try:
            mat_data = scipy.io.loadmat(str(file_path))
            
            if 'data' in mat_data:
                meg_data = mat_data['data']
            elif 'B1' in mat_data:
                meg_data = mat_data['B1']
            else:
                data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                if data_keys:
                    meg_data = mat_data[data_keys[0]]
                else:
                    raise ValueError("无法找到数据键")
            
            if meg_data.shape[0] > meg_data.shape[1]:
                meg_data = meg_data.T
                
            if meg_data.shape[0] > self.n_channels:
                meg_data = meg_data[:self.n_channels, :]
            
            # fT量级数据处理
            if np.max(np.abs(meg_data)) > 0:
                scale_factor = 1e12
                meg_data = meg_data * scale_factor
            
            # 填充或截断
            channels, current_len = meg_data.shape
            output = np.zeros((channels, self.max_length), dtype=np.float32)
            
            if current_len >= self.max_length:
                output = meg_data[:, :self.max_length]
            else:
                output[:, :current_len] = meg_data
                
            return torch.from_numpy(output)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return torch.zeros(self.n_channels, self.max_length, dtype=torch.float32)
    
    def __getitem__(self, index):
        # 加载数据
        x_clean = self.load_meg_sample(self.clean_files[index])
        x_noisy = self.load_meg_sample(self.noisy_files[index])
        
        # 数据增强
        if self.use_augmentation:
            x_noisy = self._augment_signal(x_noisy)
        
        # 子采样
        g1_meg, g2_meg = subsample2_meg(x_noisy)
        
        return x_noisy, g1_meg, x_clean, g2_meg, x_noisy 