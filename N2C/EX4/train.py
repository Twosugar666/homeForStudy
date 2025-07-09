#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEG信号Conformer去噪训练脚本
"""

import os
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from typing import Dict

from dataset import MEGDenoiseDataset, create_data_loaders
from conformer_denoiser import ConformerDenoiser, create_conformer_model
from loss import create_loss_function, compute_snr, compute_pearson_correlation


def setup_logging(log_dir: str) -> logging.Logger:
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def save_config(config: dict, save_dir: str):
    """保存配置文件"""
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience


class Trainer:
    """训练器类"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        self.output_dir = config['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置日志
        self.logger = setup_logging(self.output_dir)
        self.logger.info(f"使用设备: {self.device}")
        
        # 保存配置
        save_config(config, self.output_dir)
        
        # 创建TensorBoard写入器
        self.writer = SummaryWriter(os.path.join(self.output_dir, 'tensorboard'))
        
        # 创建模型
        self.model = create_conformer_model(config['model']).to(self.device)
        self.logger.info(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 创建损失函数
        self.criterion = create_loss_function(config.get('loss', {'type': 'mse'}))
        
        # 创建优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['max_epochs'],
            eta_min=config['training']['learning_rate'] * 0.01
        )
        
        # 早停机制
        self.early_stopping = EarlyStopping(
            patience=config['training']['patience'],
            min_delta=config['training']['min_delta']
        )
        
        # 梯度裁剪
        self.max_grad_norm = config['training'].get('max_grad_norm', 1.0)
        
        # 训练状态
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
    def create_data_loaders(self):
        """创建数据加载器"""
        train_data_dir = self.config['data']['train_dir']
        test_data_dir = self.config['data'].get('test_dir', None)
        n2n_target_dir = self.config['data'].get('n2n_target_dir', None)
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = create_data_loaders(
            train_data_dir=train_data_dir,
            test_data_dir=test_data_dir,
            n2n_target_dir=n2n_target_dir,
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['data']['num_workers'],
            val_split=self.config['data'].get('val_split', 0.2),
            **self.config['data'].get('dataset_kwargs', {})
        )
        
        self.logger.info(f"训练数据批次数: {len(train_loader)}")
        self.logger.info(f"验证数据批次数: {len(val_loader)}")
        if test_loader:
            self.logger.info(f"测试数据批次数: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {self.epoch+1}')
        
        for batch_idx, (noisy_data, clean_data) in enumerate(progress_bar):
            # 移动数据到设备
            noisy_data = noisy_data.to(self.device)
            clean_data = clean_data.to(self.device)
            
            # 前向传播
            predictions = self.model(noisy_data)
            
            # 计算损失
            if hasattr(self.criterion, 'forward') and 'dict' in str(type(self.criterion(predictions, clean_data))):
                # 组合损失函数返回字典
                loss_dict = self.criterion(predictions, clean_data)
                loss = loss_dict['total']
            else:
                # 简单损失函数
                loss = self.criterion(predictions, clean_data)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # 更新参数
            self.optimizer.step()
            
            # 记录损失
            total_loss += loss.item()
            self.global_step += 1
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.6f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # 记录到TensorBoard
            if self.global_step % self.config['logging']['log_interval'] == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/learning_rate', 
                                     self.optimizer.param_groups[0]['lr'], self.global_step)
                
        return total_loss / num_batches
        
    def validate_epoch(self, val_loader: DataLoader) -> dict:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        total_snr = 0.0
        total_corr = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for noisy_data, clean_data in tqdm(val_loader, desc='Validation'):
                # 移动数据到设备
                noisy_data = noisy_data.to(self.device)
                clean_data = clean_data.to(self.device)
                
                # 前向传播
                predictions = self.model(noisy_data)
                
                # 计算损失
                if hasattr(self.criterion, 'forward') and 'dict' in str(type(self.criterion(predictions, clean_data))):
                    loss_dict = self.criterion(predictions, clean_data)
                    loss = loss_dict['total']
                else:
                    loss = self.criterion(predictions, clean_data)
                
                total_loss += loss.item()
                
                # 计算评估指标
                snr = compute_snr(predictions, clean_data)
                correlation = compute_pearson_correlation(predictions, clean_data)
                
                total_snr += snr.mean().item()
                total_corr += correlation.mean().item()
        
        metrics = {
            'loss': total_loss / num_batches,
            'snr_db': total_snr / num_batches,
            'correlation': total_corr / num_batches
        }
        
        return metrics
        
    def save_checkpoint(self, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.output_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 如果是最佳模型，保存为best_model.pth
        if is_best:
            best_path = os.path.join(self.output_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"保存最佳模型到: {best_path}")
            
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        
        self.logger.info(f"从检查点恢复训练: epoch {self.epoch}, step {self.global_step}")
        
    def train(self):
        """主训练循环"""
        self.logger.info("开始训练...")
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = self.create_data_loaders()
        
        for epoch in range(self.epoch, self.config['training']['max_epochs']):
            self.epoch = epoch
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            
            # 验证
            val_metrics = self.validate_epoch(val_loader)
            val_loss = val_metrics['loss']
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录日志
            self.logger.info(
                f"Epoch {epoch+1}/{self.config['training']['max_epochs']} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                f"Val SNR: {val_metrics['snr_db']:.2f} dB, "
                f"Val Corr: {val_metrics['correlation']:.3f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )
            
            # 记录到TensorBoard
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
            self.writer.add_scalar('epoch/val_snr', val_metrics['snr_db'], epoch)
            self.writer.add_scalar('epoch/val_correlation', val_metrics['correlation'], epoch)
            
            # 保存检查点
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
                
            self.save_checkpoint(is_best)
            
            # 早停检查
            if self.early_stopping(val_loss):
                self.logger.info(f"早停触发，在epoch {epoch+1}停止训练")
                break
                
        self.logger.info("训练完成！")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='MEG信号Conformer去噪训练')
    parser.add_argument('--config', type=str, default='config.json',
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--test_dir', type=str, help='Override test directory')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    parser.add_argument('--epochs', type=int, help='Override number of epochs')
    parser.add_argument('--lr', type=float, help='Override learning rate')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--n2n_target_dir', type=str, help='(N2N) Override N2N target directory')

    args = parser.parse_args()
    
    # 默认配置
    default_config = {
        'data': {
            'train_dir': 'traindata_eptr1',
            'test_dir': 'testdata_eptr2',
            'val_split': 0.2,
            'num_workers': 0,  # Windows建议设为0
            'dataset_kwargs': {
                'channels': 26,
                'time_length': 1000,
                'normalize_method': 'zscore',
                'augmentation': True,
                'data_scale_factor': 1e15
            }
        },
        'model': {
            'input_channels': 26,
            'embed_dim': 256,
            'num_layers': 6,
            'num_heads': 8,
            'ffn_dim': 1024,
            'conv_kernel_size': 31,
            'dropout': 0.1,
            'use_pos_encoding': True
        },
        'loss': {
            'type': 'combined',
            'kwargs': {
                'mse_weight': 1.0,
                'snr_weight': 0.1,
                'perceptual_weight': 0.05
            }
        },
        'training': {
            'batch_size': 16,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'max_epochs': 100,
            'patience': 15,
            'min_delta': 1e-6,
            'max_grad_norm': 1.0
        },
        'logging': {
            'log_interval': 50
        },
        'output_dir': f'runs/meg_conformer_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    }
    
    # 如果存在配置文件则加载
    if os.path.exists(args.config):
        config = load_config(args.config)
        # 合并默认配置
        for key in default_config:
            if key not in config:
                config[key] = default_config[key]
    else:
        config = default_config
        # 保存默认配置
        save_config(config, '.')
        print(f"已创建默认配置文件: {args.config}")
    
    if args.test_dir:
        config['data']['test_dir'] = args.test_dir
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.epochs:
        config['training']['max_epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.n2n_target_dir:
        config['data']['n2n_target_dir'] = args.n2n_target_dir

    # 创建训练器
    trainer = Trainer(config)
    
    # 如果需要恢复训练
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 开始训练
    trainer.train()


if __name__ == "__main__":
    main() 