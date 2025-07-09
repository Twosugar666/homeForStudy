#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEG信号去噪数据加载与预处理模块 - N2N专用版本
"""

import os
import glob
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional, Dict
import logging
from pathlib import Path


class MEGDenoiseDataset(Dataset):
    """MEG去噪数据集类 - N2N专用版本"""
    
    def __init__(
        self,
        input_data_dir: str,
        target_data_dir: str,
        transform: Optional[callable] = None,
        channels: int = 26,
        time_length: Optional[int] = None,
        augmentation: bool = False,
        normalize_method: str = 'zscore',
        data_scale_factor: float = 1e15,  # 将fT量级数据放大到合理范围
    ):
        """
        初始化MEG去噪数据集 - N2N模式
        
        Args:
            input_data_dir: 输入噪声数据目录路径，应包含noisy子文件夹
            target_data_dir: 目标噪声数据目录路径，应包含noisy子文件夹
            transform: 数据变换函数
            channels: MEG通道数，默认26
            time_length: 时间序列长度，None表示使用原始长度
            augmentation: 是否启用数据增强
            normalize_method: 标准化方法 ('zscore', 'minmax', 'none')
            data_scale_factor: 数据缩放因子，用于将fT量级数据缩放到合理范围
        """
        self.input_data_dir = Path(input_data_dir)
        self.target_data_dir = Path(target_data_dir)
        self.transform = transform
        self.channels = channels
        self.time_length = time_length
        self.augmentation = augmentation
        self.normalize_method = normalize_method
        self.data_scale_factor = data_scale_factor
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 加载数据文件列表
        self.input_files, self.target_files = self._load_file_lists()
        
        # 验证数据文件匹配
        self._validate_file_pairs()
        
        # 初始化标准化器
        self.scaler = None
        if self.normalize_method != 'none':
            self._compute_normalization_stats()
        
        self.logger.info(f"N2N数据集初始化完成: {len(self.input_files)} 个样本")
    
    def _load_file_lists(self) -> Tuple[List[str], List[str]]:
        """加载输入和目标噪声文件列表"""
        # 处理相对路径
        input_dir = self.input_data_dir
        target_dir = self.target_data_dir
        
        if not input_dir.is_absolute():
            script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            input_dir = script_dir / input_dir
            
        if not target_dir.is_absolute():
            script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
            target_dir = script_dir / target_dir

        # 输入数据来自 input_data_dir/noisy
        input_noisy_dir = input_dir / 'noisy'
        self.logger.info(f"加载输入数据目录: {input_noisy_dir}")
        if not input_noisy_dir.exists():
            raise ValueError(f"输入数据目录 {input_noisy_dir} 不存在")
        input_files = sorted(glob.glob(str(input_noisy_dir / "*.mat")))

        # 目标数据来自 target_data_dir/noisy
        target_noisy_dir = target_dir / 'noisy'
        self.logger.info(f"加载目标数据目录: {target_noisy_dir}")
        if not target_noisy_dir.exists():
            raise ValueError(f"目标数据目录 {target_noisy_dir} 不存在")
        target_files = sorted(glob.glob(str(target_noisy_dir / "*.mat")))

        if len(input_files) == 0:
            raise ValueError(f"输入目录 {input_noisy_dir} 未找到.mat数据文件")
        if len(target_files) == 0:
            raise ValueError(f"目标目录 {target_noisy_dir} 未找到.mat数据文件")
        
        return input_files, target_files
    
    def _validate_file_pairs(self):
        """验证输入和目标文件是否一一对应"""
        if len(self.input_files) != len(self.target_files):
            raise ValueError(
                f"目标文件数量({len(self.target_files)}) "
                f"与输入文件数量({len(self.input_files)})不匹配"
            )
        
        # 检查文件名是否对应
        for input_file, target_file in zip(self.input_files, self.target_files):
            input_name = Path(input_file).stem.replace('noisy_', '')
            target_name = Path(target_file).stem.replace('noisy_', '')
            
            if input_name != target_name:
                raise ValueError(f"文件名不匹配: {input_file} vs {target_file}")
    
    def _load_mat_file(self, file_path: str) -> np.ndarray:
        """
        加载.mat文件并返回标准化格式的数据
        
        Args:
            file_path: .mat文件路径
            
        Returns:
            data: shape为(channels, time)的数据数组
        """
        try:
            mat_data = scipy.io.loadmat(file_path)
            
            # 尝试不同的可能键名
            possible_keys = ['data', 'signal', 'meg_data', 'eeg_data']
            data = None
            
            for key in possible_keys:
                if key in mat_data:
                    data = mat_data[key]
                    break
            
            if data is None:
                # 如果没有找到标准键名，尝试找到最大的数组
                array_keys = [k for k, v in mat_data.items() 
                             if isinstance(v, np.ndarray) and not k.startswith('__')]
                
                if array_keys:
                    data = mat_data[array_keys[0]]
                else:
                    raise ValueError(f"无法在{file_path}中找到有效的数据数组")
            
            # 确保数据是numpy数组
            data = np.array(data, dtype=np.float32)
            
            # 数据缩放 - 将fT量级数据放大到合理范围
            data = data * self.data_scale_factor
            
            # 确保格式为(channels, time)
            if data.shape[0] > data.shape[1]:
                data = data.T
            
            # 验证通道数
            if data.shape[0] != self.channels:
                if data.shape[1] == self.channels:
                    data = data.T
                else:
                    self.logger.warning(
                        f"文件 {file_path} 的通道数({data.shape[0]}) "
                        f"与期望的通道数({self.channels})不符"
                    )
            
            return data
            
        except Exception as e:
            raise RuntimeError(f"加载文件 {file_path} 时出错: {str(e)}")
    
    def _compute_normalization_stats(self):
        """计算数据集的标准化统计量 - 基于输入噪声数据"""
        self.logger.info("N2N模式：基于输入噪声数据计算标准化统计量")
        
        all_data = []
        
        # 从前100个样本计算统计量以提高效率
        sample_files = self.input_files[:min(100, len(self.input_files))]
        
        for file_path in sample_files:
            try:
                data = self._load_mat_file(file_path)
                all_data.append(data.flatten())
            except Exception as e:
                self.logger.warning(f"计算统计量时跳过文件 {file_path}: {str(e)}")
                continue
        
        if not all_data:
            raise RuntimeError("无法加载任何数据文件来计算标准化统计量")
        
        # 合并所有数据
        all_data = np.concatenate(all_data)
        
        # 创建标准化器
        if self.normalize_method == 'zscore':
            self.scaler = StandardScaler()
            self.scaler.fit(all_data.reshape(-1, 1))
        elif self.normalize_method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler(feature_range=(-1, 1))
            self.scaler.fit(all_data.reshape(-1, 1))
        
        self.logger.info("标准化统计量计算完成")
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """对数据进行标准化"""
        if self.scaler is None:
            return data
        
        original_shape = data.shape
        data_flat = data.flatten().reshape(-1, 1)
        data_normalized = self.scaler.transform(data_flat)
        
        return data_normalized.reshape(original_shape)
    
    def _preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """
        数据预处理
        
        Args:
            data: 输入数据，shape为(channels, time)
            
        Returns:
            processed_data: 预处理后的数据，shape为(time, channels)
        """
        # 标准化
        data = self._normalize_data(data)
        
        # 截取或填充到指定长度
        if self.time_length is not None:
            if data.shape[1] > self.time_length:
                # 随机截取片段
                start_idx = np.random.randint(0, data.shape[1] - self.time_length + 1)
                data = data[:, start_idx:start_idx + self.time_length]
            elif data.shape[1] < self.time_length:
                # 零填充
                pad_width = ((0, 0), (0, self.time_length - data.shape[1]))
                data = np.pad(data, pad_width, mode='constant')
        
        # 转置为(time, channels)格式
        data = data.T
        
        # 数据增强
        if self.augmentation:
            data = self._apply_augmentation(data)
        
        return data
    
    def _apply_augmentation(self, data: np.ndarray) -> np.ndarray:
        """应用数据增强"""
        # 随机添加小量噪声
        if np.random.rand() < 0.3:
            noise_scale = 0.01 * np.std(data)
            data += np.random.normal(0, noise_scale, data.shape)
        
        # 随机时间翻转
        if np.random.rand() < 0.2:
            data = np.flip(data, axis=0)
        
        # 随机通道置换（保持大部分通道不变）
        if np.random.rand() < 0.1:
            num_swap = np.random.randint(1, min(3, data.shape[1]))
            channels = np.arange(data.shape[1])
            np.random.shuffle(channels)
            swap_idx = channels[:num_swap]
            data[:, swap_idx] = data[:, np.random.permutation(swap_idx)]
        
        return data
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.input_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一个数据样本
        
        Args:
            idx: 样本索引
            
        Returns:
            input_tensor: 输入噪声信号张量
            target_tensor: 目标噪声信号张量
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("索引超出范围")

        input_file = self.input_files[idx]
        target_file = self.target_files[idx]
        
        # 加载数据
        try:
            input_data = self._load_mat_file(input_file)
            target_data = self._load_mat_file(target_file)
        except Exception as e:
            self.logger.error(f"加载文件失败 (index {idx}): {str(e)}")
            # 返回空张量或进行其他错误处理
            return torch.zeros(self.channels, 1), torch.zeros(self.channels, 1)

        # 预处理
        input_data = self._preprocess_data(input_data)
        target_data = self._preprocess_data(target_data)

        # 转换为Tensor
        input_tensor = torch.from_numpy(input_data.copy())
        target_tensor = torch.from_numpy(target_data.copy())

        # 应用数据变换
        if self.transform:
            input_tensor, target_tensor = self.transform(input_tensor, target_tensor)
            
        return input_tensor, target_tensor
        
    def get_data_info(self) -> Dict:
        """获取数据集信息"""
        sample_data = self._load_mat_file(self.input_files[0])
        
        return {
            "num_samples": len(self),
            "channels": self.channels,
            "time_length": self.time_length,
            "augmentation": self.augmentation,
            "normalization": self.normalize_method,
            "mode": "N2N"
        }


def create_data_loaders(
    input_data_dir: str,
    target_data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_shuffle: bool = True,
    val_split: float = 0.2,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练、验证数据加载器 - N2N专用版本
    
    Args:
        input_data_dir: 输入噪声数据目录
        target_data_dir: 目标噪声数据目录
        batch_size: 批大小
        num_workers: 数据加载工作进程数
        train_shuffle: 是否打乱训练数据
        val_split: 验证集划分比例
        dataset_kwargs: 传递给MEGDenoiseDataset的其他参数
    
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    """
    
    # 训练和验证数据集
    dataset = MEGDenoiseDataset(
        input_data_dir=input_data_dir, 
        target_data_dir=target_data_dir, 
        **dataset_kwargs
    )
    
    # 划分训练集和验证集
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    if train_size == 0 or val_size == 0:
        raise ValueError("数据集太小, 无法划分为训练集和验证集")
        
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # 创建训练和验证数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
        
    return train_loader, val_loader


# 设置日志
logging.basicConfig(level=logging.INFO)

def test_n2n_dataset():
    """测试N2N数据集"""
    try:
        print("=== 测试N2N数据集加载 ===")
        
        # 测试数据集创建
        dataset = MEGDenoiseDataset(
            input_data_dir="traindata_eptr1",
            target_data_dir="traindata_whq",
            channels=26,
            time_length=1000,
            normalize_method='zscore',
            augmentation=False
        )
        
        print(f"数据集大小: {len(dataset)}")
        
        # 测试数据加载
        input_data, target_data = dataset[0]
        print(f"样本形状: input={input_data.shape}, target={target_data.shape}")
        print(f"数据范围: input=[{input_data.min():.3f}, {input_data.max():.3f}], target=[{target_data.min():.3f}, {target_data.max():.3f}]")
        
        # 测试数据加载器
        print("\n=== 测试数据加载器 ===")
        train_loader, val_loader = create_data_loaders(
            input_data_dir="traindata_eptr1",
            target_data_dir="traindata_whq",
            batch_size=8,
            num_workers=0,
            channels=26,
            time_length=1000,
            normalize_method='zscore',
            augmentation=False
        )
        
        print(f"训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader)}")
        
        # 测试一个批次
        for batch_input, batch_target in train_loader:
            print(f"批次形状: input={batch_input.shape}, target={batch_target.shape}")
            print(f"批次数据范围: input=[{batch_input.min():.3f}, {batch_input.max():.3f}], target=[{batch_target.min():.3f}, {batch_target.max():.3f}]")
            break
            
        print("\n✅ N2N数据集测试成功！")
        return True
        
    except Exception as e:
        print(f"\n❌ N2N数据集测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_n2n_dataset() 