#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEG去噪训练脚本
专门处理fT量级的MEG信号
使用成对的子采样信号进行自监督训练
"""

import os
import gc
import torch
import warnings
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
import shutil

from MEG_dataset import MEGDataset, subsample2_meg
from MEG_model import SimpleMEGDenoiser, MEGDenoiser, AttentionMEGDenoiser
from MEG_loss import SimpleMEGLoss, MEGLoss, SelfSupervisedMEGLoss

# 检查GPU可用性
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    print('✅ 在GPU上训练')
else:
    print('⚠️  在CPU上训练（速度较慢）')
DEVICE = torch.device('cuda' if train_on_gpu else 'cpu')

warnings.filterwarnings(action='ignore', category=DeprecationWarning)
np.random.seed(999)
torch.manual_seed(999)

# CUDA确定性设置
if train_on_gpu:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

############################################## 训练参数 ##############################################
BATCH_SIZE = 16  # 可以使用更大的batch size
LEARNING_RATE = 1e-4
EPOCHS = 200  # 训练通常收敛更快
SAVE_EVERY = 10  # 每5个epoch保存一次模型
SIGNAL_LENGTH = 1001  # MEG信号长度

def prepare_data():
    """准备训练和验证数据"""
    print("=== Preparing Training Data ===")
    
    # 获取数据文件路径
    all_noisy_files = sorted(list(Path('./all_whq/whq_train/noisy').rglob('*.mat')))
    all_clean_files = sorted(list(Path('./all_whq/whq_train/clean').rglob('*.mat')))
    
    print(f"Found {len(all_noisy_files)} noisy files")
    print(f"Found {len(all_clean_files)} clean files")
    
    # 使用所有训练数据
    train_noisy = all_noisy_files
    train_clean = all_clean_files
    
    # 获取验证数据
    val_noisy_files = sorted(list(Path('./all_whq/whq_valid/noisy').rglob('*.mat')))
    val_clean_files = sorted(list(Path('./all_whq/whq_valid/clean').rglob('*.mat')))
    
    print(f"Found {len(val_noisy_files)} validation noisy files")
    print(f"Found {len(val_clean_files)} validation clean files")
    
    # 创建数据集
    train_dataset = MEGDataset(train_noisy, train_clean, max_length=SIGNAL_LENGTH)
    val_dataset = MEGDataset(val_noisy_files, val_clean_files, max_length=SIGNAL_LENGTH)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader

def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, batch in enumerate(pbar):
        try:
            # 解包时域数据
            x_noisy, g1_meg, x_clean, g2_meg, _ = batch
            
            # 移动到设备
            x_noisy = x_noisy.to(DEVICE)
            g1_meg = g1_meg.to(DEVICE)
            g2_meg = g2_meg.to(DEVICE)
            
            # 前向传播
            optimizer.zero_grad()
            
            # 对第一个子采样信号进行去噪
            fg1_meg = model(g1_meg)
            
            # 对完整噪声信号去噪并进行子采样
            fx_meg = model(x_noisy)
            g1fx_list, g2fx_list = [], []
            for i in range(fx_meg.shape[0]):
                g1fx_sample, g2fx_sample = subsample2_meg(fx_meg[i].detach().cpu())
                g1fx_list.append(g1fx_sample)
                g2fx_list.append(g2fx_sample)
            g1_fy = torch.stack(g1fx_list).to(DEVICE)
            g2_fy = torch.stack(g2fx_list).to(DEVICE)
            
            # 计算自监督损失
            loss = criterion(g1_meg, fg1_meg, g2_meg, g1_fy, g2_fy)
            
            # 检查损失是否为NaN
            if torch.isnan(loss):
                print(f"Warning: NaN loss in batch {batch_idx}, skipping")
                continue
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg_Loss': f'{total_loss/(batch_idx+1):.6f}'
            })
            
            # 内存清理
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache() if train_on_gpu else None
                
        except Exception as e:
            print(f"Failed to train batch {batch_idx}: {e}")
            continue
    
    return total_loss / len(train_loader) if len(train_loader) > 0 else float('inf')

def validate(model, val_loader, criterion):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            try:
                x_noisy, g1_meg, x_clean, g2_meg, _ = batch
                
                x_noisy = x_noisy.to(DEVICE)
                g1_meg = g1_meg.to(DEVICE)
                g2_meg = g2_meg.to(DEVICE)
                
                # 前向传播
                fg1_meg = model(g1_meg)
                
                # 损失计算
                fx_meg = model(x_noisy)
                # 处理批次数据的子采样 (分离梯度)
                g1fx_list, g2fx_list = [], []
                for i in range(fx_meg.shape[0]):
                    g1fx_sample, g2fx_sample = subsample2_meg(fx_meg[i].detach().cpu())
                    g1fx_list.append(g1fx_sample)
                    g2fx_list.append(g2fx_sample)
                g1fx = torch.stack(g1fx_list).to(DEVICE)
                g2fx = torch.stack(g2fx_list).to(DEVICE)
                
                loss = criterion(g1_meg, fg1_meg, g2_meg, g1fx, g2fx)
                
                if not torch.isnan(loss):
                    total_loss += loss.item()
                
            except Exception as e:
                print(f"Validation batch {batch_idx} failed: {e}")
                continue
    
    return total_loss / len(val_loader) if len(val_loader) > 0 else float('inf')

def visualize_denoising_results(model, val_loader, save_dir):
    """可视化去噪效果"""
    print("\n=== Visualizing Denoising Results ===")
    model.eval()
    
    # 获取一个批次的数据
    for batch in val_loader:
        x_noisy, _, x_clean, _, _ = batch
        break
    
    # 选择一个样本进行可视化
    sample_idx = 0
    noisy_signal = x_noisy[sample_idx].cpu().numpy()
    clean_signal = x_clean[sample_idx].cpu().numpy()
    
    # 模型去噪
    with torch.no_grad():
        denoised_tensor = model(x_noisy.to(DEVICE))
    
    denoised_signal = denoised_tensor[sample_idx].cpu().numpy()
    
    # 绘图
    plt.figure(figsize=(12, 8))
    
    # 时域可视化
    plt.subplot(3, 1, 1)
    plt.plot(clean_signal[0])
    plt.title('Original Clean MEG Signal')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(noisy_signal[0])
    plt.title('Noisy MEG Signal')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(denoised_signal[0])
    plt.title('Denoised MEG Signal')
    plt.xlabel('Time Points')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'time_domain_denoising_result.png')
    print(f"✅ Time domain denoising result saved to {save_dir / 'time_domain_denoising_result.png'}")
    
    # 计算信噪比改善
    def calculate_snr(clean, noisy):
        noise = noisy - clean
        signal_power = np.mean(clean ** 2)
        noise_power = np.mean(noise ** 2)
        return 10 * np.log10(signal_power / noise_power)
    
    original_snr = calculate_snr(clean_signal, noisy_signal)
    denoised_snr = calculate_snr(clean_signal, denoised_signal)
    
    snr_improvement = denoised_snr - original_snr
    
    print(f"Original SNR: {original_snr:.2f} dB")
    print(f"Denoised SNR: {denoised_snr:.2f} dB")
    print(f"SNR Improvement: {snr_improvement:.2f} dB")
    
    return denoised_signal

def plot_training_history(train_losses, val_losses, save_dir):
    """绘制训练历史"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir / 'training_history.png')
    print(f"✅ Training history plot saved to {save_dir / 'training_history.png'}")

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, save_dir):
    """训练模型"""
    print("\n=== Starting Training ===")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # 训练
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        train_losses.append(train_loss)
        
        # 验证
        val_loss = validate(model, val_loader, criterion)
        val_losses.append(val_loss)
        
        # 调整学习率
        scheduler.step(val_loss)
        
        # 打印结果
        print(f"Training Loss: {train_loss:.6f}")
        print(f"Validation Loss: {val_loss:.6f}")
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 绘制当前训练进度
        if (epoch + 1) % 5 == 0:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training Progress (Epoch {epoch+1}/{EPOCHS})')
            plt.legend()
            plt.grid(True)
            plt.savefig(save_dir / f'training_progress_epoch_{epoch+1}.png')
            plt.close()
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'model_type': 'AttentionMEGDenoiser'
            }, save_dir / 'best_model.pth')
            print(f"✅ Saved best model (Validation Loss: {val_loss:.6f})")
        
        # 定期保存检查点
        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'model_type': 'AttentionMEGDenoiser'
            }, save_dir / f'checkpoint_epoch_{epoch+1}.pth')
            print(f"💾 Saved checkpoint: epoch_{epoch+1}")
        
        # 内存清理
        gc.collect()
        if train_on_gpu:
            torch.cuda.empty_cache()
    
    print("\n🎉 Training Complete!")
    print(f"Best Validation Loss: {best_val_loss:.6f}")
    print(f"Models saved to: {save_dir}")
    
    # 绘制完整训练历史
    plot_training_history(train_losses, val_losses, save_dir)
    
    # 加载最佳模型进行可视化
    best_model = AttentionMEGDenoiser(n_channels=1, signal_length=SIGNAL_LENGTH).to(DEVICE)
    checkpoint = torch.load(save_dir / 'best_model.pth')
    best_model.load_state_dict(checkpoint['model_state_dict'])
    
    # 保存一个样本结果
    denoised_signal = visualize_denoising_results(best_model, val_loader, save_dir)
    
    result_data = {
        'denoised': denoised_signal,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': checkpoint['epoch'],
        'best_val_loss': best_val_loss
    }
    np.save(save_dir / 'denoising_result.npy', result_data)
    
    # 绘制最终结果
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.axvline(x=checkpoint['epoch'], color='r', linestyle='--', label=f'Best Model (Epoch {checkpoint["epoch"]+1})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    for i in range(min(3, denoised_signal.shape[0])):  
        plt.plot(denoised_signal[i], label=f'Channel {i+1}')
    plt.xlabel('Time Points')
    plt.ylabel('Amplitude')
    plt.title('Denoised MEG Signal')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'denoising_result.png')
    print(f"✅ Final result plot saved to {save_dir / 'denoising_result.png'}")
    
    return best_model

def main():
    """主函数"""
    print("\n🚀 Starting Self-Supervised MEG Denoising Model Training")
    
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("./output/checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = output_dir / f"results_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Results will be saved to: {results_dir}")
    
    # 准备数据
    train_loader, val_loader = prepare_data()
    
    # 创建模型
    print("\n=== Creating Model ===")
    model = AttentionMEGDenoiser(n_channels=1, signal_length=SIGNAL_LENGTH).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 使用新的自监督损失函数
    criterion = SelfSupervisedMEGLoss(gamma=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # 训练模型
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, results_dir)
    
    print("\n✅ Training completed!")

if __name__ == "__main__":
    main() 