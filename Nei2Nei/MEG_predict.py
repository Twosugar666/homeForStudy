#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEG去噪预测脚本
使用训练好的模型对测试集进行预测并保存结果
"""

import os
import torch
import numpy as np
import scipy.io as sio
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from MEG_model import SimpleMEGDenoiser, MEGDenoiser, AttentionMEGDenoiser

def predict_and_save(model_path, output_dir='./denoised_results'):
    """使用训练好的模型对测试集进行预测并保存结果"""
    print("\n=== Predicting and Saving Denoised Results ===")
    
    # 确保输出目录存在
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # 根据checkpoint中的model_type决定使用哪种模型
    model_type = checkpoint.get('model_type', 'SimpleMEGDenoiser')
    
    if 'Attention' in model_type:
        model = AttentionMEGDenoiser(n_channels=1, signal_length=1001)
    elif 'MEGDenoiser' in model_type:
        model = MEGDenoiser(n_channels=1, signal_length=1001)
    else:
        model = SimpleMEGDenoiser(n_channels=1, signal_length=1001)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully: {model_type}")
    
    # 获取测试数据文件路径
    test_files = sorted(list(Path('./all_whq/whq_test/noisy').rglob('*.mat')))
    print(f"Found {len(test_files)} test files")
    
    # 处理每个测试文件
    for file_path in tqdm(test_files, desc="Processing test files"):
        try:
            # 加载MAT文件
            mat_data = sio.loadmat(str(file_path))
            
            # 确定数据键
            data_key = None
            for key in ['data', 'B1']:
                if key in mat_data:
                    data_key = key
                    break
            
            if data_key is None:
                print(f"Warning: Could not find data key in {file_path}")
                continue
            
            # 获取数据
            meg_data = mat_data[data_key]
            
            # 确保数据形状为 [channels, time_points]
            if meg_data.shape[0] > meg_data.shape[1]:
                meg_data = meg_data.T
            
            # 数值稳定性处理
            if np.max(np.abs(meg_data)) > 0:
                scale_factor = 1e12  # 将fT缩放到pT量级
                meg_data = meg_data * scale_factor
            
            # 转换为tensor
            meg_tensor = torch.from_numpy(meg_data).float().unsqueeze(0)
            
            # 模型预测
            with torch.no_grad():
                denoised_tensor = model(meg_tensor.to(device))
            
            # 转回numpy并还原缩放
            denoised_data = denoised_tensor.cpu().numpy().squeeze(0)
            if np.max(np.abs(meg_data)) > 0:
                denoised_data = denoised_data / scale_factor
            
            # 创建输出文件名
            output_filename = file_path.stem + "_denoised.mat"
            output_file_path = output_path / output_filename
            
            # 保存为MAT文件
            sio.savemat(str(output_file_path), {data_key: denoised_data})
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"✅ Denoised results saved to {output_dir}")
    return True

def visualize_sample_result(model_path, sample_idx=0):
    """可视化一个样本的去噪效果"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    model_type = checkpoint.get('model_type', 'SimpleMEGDenoiser')
    
    if 'Attention' in model_type:
        model = AttentionMEGDenoiser(n_channels=1, signal_length=1001)
    elif 'MEGDenoiser' in model_type:
        model = MEGDenoiser(n_channels=1, signal_length=1001)
    else:
        model = SimpleMEGDenoiser(n_channels=1, signal_length=1001)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 获取测试数据
    test_noisy_files = sorted(list(Path('./all_whq/whq_test/noisy').rglob('*.mat')))
    test_clean_files = sorted(list(Path('./all_whq/whq_test/clean').rglob('*.mat')))
    
    if sample_idx >= len(test_noisy_files):
        print(f"Sample index {sample_idx} out of range. Using index 0.")
        sample_idx = 0
    
    # 加载样本
    noisy_file = test_noisy_files[sample_idx]
    clean_file = test_clean_files[sample_idx]
    
    # 加载MAT文件
    noisy_mat = sio.loadmat(str(noisy_file))
    clean_mat = sio.loadmat(str(clean_file))
    
    # 确定数据键
    data_key = None
    for key in ['data', 'B1']:
        if key in noisy_mat:
            data_key = key
            break
    
    if data_key is None:
        print("Warning: Could not find data key in files")
        return
    
    # 获取数据
    noisy_data = noisy_mat[data_key]
    clean_data = clean_mat[data_key]
    
    # 确保数据形状为 [channels, time_points]
    if noisy_data.shape[0] > noisy_data.shape[1]:
        noisy_data = noisy_data.T
    if clean_data.shape[0] > clean_data.shape[1]:
        clean_data = clean_data.T
    
    # 数值稳定性处理
    scale_factor = 1.0
    if np.max(np.abs(noisy_data)) > 0:
        scale_factor = 1e12  # 将fT缩放到pT量级
        noisy_data = noisy_data * scale_factor
        clean_data = clean_data * scale_factor
    
    # 转换为tensor
    noisy_tensor = torch.from_numpy(noisy_data).float().unsqueeze(0)
    
    # 模型预测
    with torch.no_grad():
        denoised_tensor = model(noisy_tensor.to(device))
    
    # 转回numpy
    denoised_data = denoised_tensor.cpu().numpy().squeeze(0)
    
    # 绘图
    plt.figure(figsize=(12, 8))
    
    # 选择第一个通道进行可视化
    channel = 0
    
    # 时域可视化
    plt.subplot(3, 1, 1)
    plt.plot(clean_data[channel])
    plt.title('Original Clean MEG Signal')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(noisy_data[channel])
    plt.title('Noisy MEG Signal')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(denoised_data[channel])
    plt.title('Denoised MEG Signal')
    plt.xlabel('Time Points')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('sample_denoising_result.png')
    plt.show()
    
    # 计算信噪比改善
    def calculate_snr(clean, noisy):
        noise = noisy - clean
        signal_power = np.mean(clean ** 2)
        noise_power = np.mean(noise ** 2)
        return 10 * np.log10(signal_power / noise_power)
    
    original_snr = calculate_snr(clean_data, noisy_data)
    denoised_snr = calculate_snr(clean_data, denoised_data)
    
    print(f"Original SNR: {original_snr:.2f} dB")
    print(f"Denoised SNR: {denoised_snr:.2f} dB")
    print(f"SNR Improvement: {denoised_snr - original_snr:.2f} dB")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='MEG Denoising Prediction')
    parser.add_argument('--model', type=str, default='F:/Nei2Nei/output/checkpoints_20250618_174800/best_model.pth', 
                        help='Path to the trained model')
    parser.add_argument('--output', type=str, default='./denoised_results',
                        help='Output directory for denoised results')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize a sample result')
    parser.add_argument('--sample', type=int, default=0,
                        help='Sample index for visualization')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found at {args.model}")
        return
    
    print(f"Using model: {args.model}")
    
    # 执行预测
    if args.visualize:
        visualize_sample_result(args.model, args.sample)
    else:
        predict_and_save(args.model, args.output)

if __name__ == "__main__":
    main() 