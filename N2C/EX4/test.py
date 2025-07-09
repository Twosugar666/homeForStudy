#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEG去噪结果拼接脚本
读取去噪后的数据，按照testdata_generate(eptr2).py脚本的方式拼接数据，
保存为.mat文件
"""

import os
import tempfile
import sys
from pathlib import Path

# 设置工作目录为脚本所在目录，确保相对路径正确
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"工作目录已设置为: {os.getcwd()}")

# 设置joblib的临时文件夹为一个只包含ASCII字符的路径
os.environ['JOBLIB_TEMP_FOLDER'] = os.path.join(tempfile.gettempdir(), 'joblib_temp')

import numpy as np
import scipy.io
import logging
import argparse
import json
import glob
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 模型相关导入
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error

from dataset import MEGDenoiseDataset, create_data_loaders
from conformer_denoiser import ConformerDenoiser, create_conformer_model
from loss import compute_snr, compute_pearson_correlation

class MEGDenoiseEvaluator:
    """MEG去噪模型评估器"""
    
    def __init__(
        self,
        model_path: str,
        config_path: str = None,
        device: str = 'auto'
    ):
        """
        初始化评估器
        
        Args:
            model_path: 模型权重文件路径
            config_path: 配置文件路径
            device: 设备类型 ('auto', 'cpu', 'cuda')
        """
        # 保存模型路径
        self.model_path = model_path
        
        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 加载配置和模型
        self.config = self._load_config(model_path, config_path)
        self.model = self._load_model(model_path)
        
        self.logger.info(f"评估器初始化完成，使用设备: {self.device}")
        
    def _load_config(self, model_path: str, config_path: Optional[str]) -> Dict:
        """加载配置文件"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # 从模型目录加载config.json
        model_dir = Path(model_path).parent
        config_file = model_dir / 'config.json'
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # 使用默认配置
        self.logger.warning("未找到配置文件，使用默认配置")
        return {
            'model': {
                'input_channels': 26,
                'embed_dim': 256,
                'num_layers': 6,
                'num_heads': 8,
                'ffn_dim': 1024,
                'conv_kernel_size': 31,
                'dropout': 0.1
            }
        }
    
    def _load_model(self, model_path: str) -> ConformerDenoiser:
        """加载模型"""
        # 创建模型
        model = create_conformer_model(self.config['model'])
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        self.logger.info(f"模型加载完成，参数数量: {model.get_num_params():,}")
        
        return model
    
    def run_denoising(
        self,
        test_dir: str,
        output_dir: str,
        batch_size: int = 32
    ) -> bool:
        """
        运行去噪过程
        
        Args:
            test_dir: 测试数据目录
            output_dir: 输出目录
            batch_size: 批大小
        
        Returns:
            bool: 是否成功
        """
        self.logger.info(f"开始去噪过程: {test_dir}")
        
        # 直接创建测试数据集
        test_dataset = MEGDenoiseDataset(
            test_dir,
            channels=self.config['model']['input_channels'],
            time_length=None,
            normalize_method='zscore',
            augmentation=False,
            data_scale_factor=1e15
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        # 评估指标
        all_mse = []
        all_mae = []
        all_snr = []
        all_correlation = []
        
        # 获取原始文件名列表
        noisy_files = sorted(test_dataset.noisy_files)
        
        # 创建保存去噪数据的目录
        denoised_dir = os.path.join(output_dir, 'denoised_data')
        os.makedirs(denoised_dir, exist_ok=True)
        self.logger.info(f"将保存去噪数据到: {denoised_dir}")
        
        file_idx = 0
        with torch.no_grad():
            for batch_idx, (noisy_data, clean_data) in enumerate(tqdm(test_loader, desc='去噪中')):
                # 移动到设备
                noisy_data = noisy_data.to(self.device)
                clean_data = clean_data.to(self.device)
                
                # 模型推理
                predictions = self.model(noisy_data)
                
                # 移动到CPU计算指标
                pred_cpu = predictions.cpu()
                clean_cpu = clean_data.cpu()
                
                # 计算指标
                for i in range(pred_cpu.shape[0]):
                    if file_idx >= len(noisy_files):
                        break
                        
                    pred_sample = pred_cpu[i].numpy()
                    clean_sample = clean_cpu[i].numpy()
                    
                    # 获取原始文件名
                    noisy_file = noisy_files[file_idx]
                    noisy_name = os.path.basename(noisy_file)
                    file_num = noisy_name.replace('noisy_', '').replace('.mat', '')
                    
                    # 保存去噪数据
                    denoised_file = os.path.join(denoised_dir, f'denoised_{file_num}')
                    # 转置回原始格式 (channels, time)
                    denoised_data = pred_sample.T
                    # 保存
                    scipy.io.savemat(denoised_file + '.mat', {'data': denoised_data})
                    
                    # MSE和MAE
                    mse = mean_squared_error(clean_sample.flatten(), pred_sample.flatten())
                    mae = mean_absolute_error(clean_sample.flatten(), pred_sample.flatten())
                    
                    all_mse.append(mse)
                    all_mae.append(mae)
                    
                    file_idx += 1
                
                # 批量计算SNR和相关系数
                snr_batch = compute_snr(predictions, clean_data)
                corr_batch = compute_pearson_correlation(predictions, clean_data)
                
                all_snr.extend(snr_batch.cpu().numpy())
                all_correlation.extend(corr_batch.mean(dim=1).cpu().numpy())  # 平均通道相关性
        
        # 计算平均指标
        results = {
            'mse': np.mean(all_mse),
            'mae': np.mean(all_mae),
            'snr_db': np.mean(all_snr),
            'correlation': np.mean(all_correlation),
            'mse_std': np.std(all_mse),
            'mae_std': np.std(all_mae),
            'snr_db_std': np.std(all_snr),
            'correlation_std': np.std(all_correlation),
            'num_samples': len(all_mse),
            'test_dir': os.path.abspath(test_dir),
            'model_path': os.path.abspath(self.model_path),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 打印结果摘要
        self._print_summary(results)
        
        # 保存基本结果
        os.makedirs(output_dir, exist_ok=True)
        results_file = os.path.join(output_dir, 'evaluation_results.json')
        
        # 将NumPy类型转换为Python原生类型
        json_results = {}
        for key, value in results.items():
            if isinstance(value, (np.float32, np.float64)):
                json_results[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                json_results[key] = int(value)
            else:
                json_results[key] = value
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2)
        self.logger.info(f"评估结果已保存到: {results_file}")
        
        return True
    
    def _print_summary(self, results: Dict):
        """打印评估结果摘要"""
        self.logger.info("评估结果摘要:")
        self.logger.info(f"样本数量: {results['num_samples']}")
        self.logger.info(f"MSE: {results['mse']:.6f} ± {results['mse_std']:.6f}")
        self.logger.info(f"MAE: {results['mae']:.6f} ± {results['mae_std']:.6f}")
        self.logger.info(f"SNR: {results['snr_db']:.2f} ± {results['snr_db_std']:.2f} dB")
        self.logger.info(f"相关系数: {results['correlation']:.4f} ± {results['correlation_std']:.4f}")


# 默认参数设置 - 使用相对路径
DEFAULT_MODEL_PATH = "runs/meg_conformer_train/best_model.pth"  # 模型权重文件的相对路径
DEFAULT_TEST_DIR = "testdata_eptr2"  # 测试数据目录的相对路径
DEFAULT_DENOISED_DIR = "visualization_results/denoised_data"  # 去噪数据目录的相对路径
DEFAULT_OUTPUT_DIR = r"F:\home\homeForStudy\N2C\EX4\visualization_results"  # 输出目录的绝对路径

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MEG去噪结果拼接')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                       help=f'模型文件路径 (.pth)，默认: {DEFAULT_MODEL_PATH}')
    parser.add_argument('--test_dir', type=str, default=DEFAULT_TEST_DIR,
                       help=f'测试数据目录，默认: {DEFAULT_TEST_DIR}')
    parser.add_argument('--denoised_dir', type=str, default=DEFAULT_DENOISED_DIR,
                       help=f'去噪数据目录，默认: {DEFAULT_DENOISED_DIR}')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                       help=f'输出目录，默认: {DEFAULT_OUTPUT_DIR}')
    parser.add_argument('--run_denoise', action='store_true',
                       help='是否先运行去噪过程')
    parser.add_argument('--force_run_denoise', action='store_true',
                       help='强制运行去噪过程，即使已存在去噪数据')
    
    # 检查命令行参数，如果没有--run_denoise，则默认添加
    if '--run_denoise' not in sys.argv and '--force_run_denoise' not in sys.argv:
        print("默认启用--run_denoise参数")
        sys.argv.append('--run_denoise')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # 确保路径是绝对路径
    model_path = os.path.abspath(args.model)
    test_dir = os.path.abspath(args.test_dir)
    denoised_dir = os.path.abspath(args.denoised_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    # 检查去噪数据目录
    denoised_files = []
    if os.path.exists(denoised_dir):
        denoised_files = sorted(glob.glob(os.path.join(denoised_dir, "denoised_*.mat")))
        if denoised_files:
            logger.info(f"找到 {len(denoised_files)} 个去噪数据文件")
    
    # 如果需要运行去噪过程或者没有找到去噪数据文件
    if args.run_denoise or args.force_run_denoise or not denoised_files:
        logger.info(f"运行去噪过程，使用模型: {model_path}")
        
        # 确保去噪数据目录存在
        denoised_parent_dir = os.path.dirname(denoised_dir)
        os.makedirs(denoised_parent_dir, exist_ok=True)
        
        # 创建评估器并运行去噪
        evaluator = MEGDenoiseEvaluator(model_path, device='auto')
        success = evaluator.run_denoising(test_dir, denoised_parent_dir)
        
        if success:
            logger.info("去噪过程完成")
            # 重新获取去噪数据文件
            denoised_files = sorted(glob.glob(os.path.join(denoised_dir, "denoised_*.mat")))
        else:
            logger.error("去噪过程失败")
            return
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logger.info(f"找到 {len(denoised_files)} 个去噪数据文件，开始拼接...")
    
    # 加载第一个文件以获取通道数
    first_data = load_mat_data(denoised_files[0])
    n_channels = first_data.shape[0] if first_data.shape[0] < first_data.shape[1] else first_data.shape[1]
    logger.info(f"检测到通道数: {n_channels}")
    
    # 创建用于拼接的数据数组
    all_data = []
    
    # 加载并拼接所有文件
    for file_path in tqdm(denoised_files, desc="拼接数据"):
        data = load_mat_data(file_path)
        
        # 确保数据格式为 [channels, time]
        if data.shape[0] > data.shape[1]:
            data = data.T
        
        all_data.append(data)
    
    # 水平拼接所有数据
    concatenated_data = np.hstack(all_data)
    logger.info(f"数据拼接完成，形状: {concatenated_data.shape}")
    
    # 保存拼接后的数据为MAT格式
    mat_path = output_dir / 'concatenated_denoised.mat'
    scipy.io.savemat(mat_path, {'data': concatenated_data})
    
    logger.info(f"拼接后的数据已保存: {mat_path}")
    
    print(f"\n数据拼接完成！")
    print(f"拼接了 {len(denoised_files)} 个去噪数据文件")
    print(f"结果保存在: {mat_path}")


def load_mat_data(file_path: str) -> np.ndarray:
    """加载.mat文件数据"""
    mat_data = scipy.io.loadmat(file_path)
    
    # 优先使用'data'键
    if 'data' in mat_data:
        data = mat_data['data']
    else:
        # 找到最大的数组作为数据
        array_keys = [k for k, v in mat_data.items() 
                     if isinstance(v, np.ndarray) and not k.startswith('__')]
        if array_keys:
            max_key = max(array_keys, key=lambda k: mat_data[k].size)
            data = mat_data[max_key]
        else:
            raise ValueError(f"无法在{file_path}中找到有效的数据数组")
    
    return np.array(data, dtype=np.float32)


if __name__ == "__main__":
    main()
