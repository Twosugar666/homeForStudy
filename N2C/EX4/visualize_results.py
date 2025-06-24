#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEG去噪结果可视化脚本
读取去噪后的数据，按照testdata_generate(eptr2).py脚本的方式拼接数据，
生成可视化图（如evoked响应对比、topomap）
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import mne
import scipy.io
from pathlib import Path
import logging
from typing import List, Tuple, Optional, Dict
import argparse
import json

from evaluate import MEGDenoiseEvaluator


class MEGVisualizationGenerator:
    """MEG去噪结果可视化生成器"""
    
    def __init__(
        self,
        model_path: str,
        config_path: str = None,
        sensor_info_path: str = r'E:\study\MEG\MEG\data\20230726zrcpos',
        output_dir: str = 'visualization_results'
    ):
        """
        初始化可视化生成器
        
        Args:
            model_path: 模型权重文件路径
            config_path: 配置文件路径
            sensor_info_path: 传感器位置信息文件路径
            output_dir: 输出目录
        """
        self.model_path = model_path
        self.config_path = config_path
        self.sensor_info_path = sensor_info_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
        # 创建评估器
        self.evaluator = MEGDenoiseEvaluator(model_path, config_path)
        
        # 加载传感器信息
        self.montage = self._load_sensor_info()
        
        self.logger.info("可视化生成器初始化完成")
    
    def _load_sensor_info(self) -> mne.channels.DigMontage:
        """加载传感器位置信息"""
        try:
            # 加载传感器位置文件
            sensor_file = scipy.io.loadmat(self.sensor_info_path)
            sensor_data = sensor_file['new_sensors']
            sensor_pos = sensor_data[0, 0]['pos']  # (104, 3)
            sensor_pos = sensor_pos.T / 1000  # (3, 104) 转换为米
            
            # 选择26个通道的传感器位置
            sensor_pick = [9, 18, 29, 5, 8, 27, 22, 17, 19, 30, 15, 11, 10, 7, 3, 0, 26, 20, 4, 24, 2, 31, 21, 23, 25, 13]
            pos_final = sensor_pos[0:3, sensor_pick]  # 提取26个通道的传感器位置
            
            # 创建通道位置字典
            ch_pos = {str(i): pos_final[0:3, i] for i in range(26)}
            
            # 创建montage
            montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
            
            self.logger.info("传感器位置信息加载成功")
            return montage
            
        except Exception as e:
            self.logger.warning(f"无法加载传感器位置信息: {e}")
            # 创建默认montage
            return self._create_default_montage()
    
    def _create_default_montage(self) -> mne.channels.DigMontage:
        """创建默认的传感器位置"""
        # 创建简单的圆形排列
        angles = np.linspace(0, 2 * np.pi, 26, endpoint=False)
        radius = 0.1  # 10cm半径
        
        ch_pos = {}
        for i in range(26):
            x = radius * np.cos(angles[i])
            y = radius * np.sin(angles[i])
            z = 0.0
            ch_pos[str(i)] = np.array([x, y, z])
        
        return mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    
    def _create_raw_from_data(
        self,
        data: np.ndarray,
        sfreq: float = 1000.0,
        ch_types: str = 'meg'
    ) -> mne.io.Raw:
        """
        从数据创建MNE Raw对象
        
        Args:
            data: 数据数组 [channels, time]
            sfreq: 采样频率
            ch_types: 通道类型
            
        Returns:
            raw: MNE Raw对象
        """
        n_channels = data.shape[0]
        
        # 创建info
        ch_names = [f'MEG{i:03d}' for i in range(n_channels)]
        info = mne.create_info(
            ch_names=ch_names,
            ch_types=ch_types,
            sfreq=sfreq
        )
        
        # 创建Raw对象
        raw = mne.io.RawArray(data, info)
        
        # 设置montage
        if self.montage is not None:
            try:
                raw.set_montage(self.montage)
            except Exception as e:
                self.logger.warning(f"设置montage失败: {e}")
        
        return raw
    
    def process_single_file(
        self,
        noisy_file: str,
        clean_file: str,
        output_prefix: str
    ) -> Dict:
        """
        处理单个文件对，生成去噪结果和可视化
        
        Args:
            noisy_file: 噪声文件路径
            clean_file: 干净文件路径
            output_prefix: 输出文件前缀
            
        Returns:
            metrics: 评估指标字典
        """
        self.logger.info(f"处理文件: {noisy_file}")
        
        # 加载数据
        noisy_data = self._load_mat_data(noisy_file)
        clean_data = self._load_mat_data(clean_file)
        
        # 确保数据格式正确 [time, channels]
        if noisy_data.shape[0] < noisy_data.shape[1]:
            noisy_data = noisy_data.T
        if clean_data.shape[0] < clean_data.shape[1]:
            clean_data = clean_data.T
        
        # 模型推理
        denoised_data, metrics = self.evaluator.evaluate_single_sample(
            noisy_data, clean_data
        )
        
        # 转换为 [channels, time] 格式用于MNE
        noisy_data_mne = noisy_data.T
        clean_data_mne = clean_data.T
        denoised_data_mne = denoised_data.T
        
        # 创建Raw对象
        raw_noisy = self._create_raw_from_data(noisy_data_mne)
        raw_clean = self._create_raw_from_data(clean_data_mne)
        raw_denoised = self._create_raw_from_data(denoised_data_mne)
        
        # 生成可视化
        self._generate_visualizations(
            raw_noisy, raw_clean, raw_denoised,
            output_prefix, metrics
        )
        
        return metrics
    
    def process_dataset(
        self,
        test_dir: str,
        max_files: int = 10
    ) -> List[Dict]:
        """
        处理整个测试数据集
        
        Args:
            test_dir: 测试数据目录
            max_files: 最大处理文件数
            
        Returns:
            all_metrics: 所有文件的评估指标列表
        """
        test_path = Path(test_dir)
        noisy_dir = test_path / 'noisy'
        clean_dir = test_path / 'clean'
        
        # 获取文件列表
        noisy_files = sorted(list(noisy_dir.glob('*.mat')))[:max_files]
        
        all_metrics = []
        
        for i, noisy_file in enumerate(noisy_files):
            # 找到对应的clean文件
            file_id = noisy_file.stem.replace('noisy_', '')
            clean_file = clean_dir / f'clean_{file_id}.mat'
            
            if not clean_file.exists():
                self.logger.warning(f"找不到对应的clean文件: {clean_file}")
                continue
            
            # 处理文件
            output_prefix = f'sample_{i+1:03d}'
            metrics = self.process_single_file(
                str(noisy_file), str(clean_file), output_prefix
            )
            
            all_metrics.append(metrics)
        
        # 生成汇总报告
        self._generate_summary_report(all_metrics)
        
        return all_metrics
    
    def _load_mat_data(self, file_path: str) -> np.ndarray:
        """加载.mat文件数据"""
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
        
        # 数据缩放
        data = np.array(data, dtype=np.float32) * 1e15
        
        return data
    
    def _generate_visualizations(
        self,
        raw_noisy: mne.io.Raw,
        raw_clean: mne.io.Raw,
        raw_denoised: mne.io.Raw,
        output_prefix: str,
        metrics: Dict
    ):
        """生成各种可视化图表"""
        
        # 1. 时间序列对比图
        self._plot_time_series_comparison(
            raw_noisy, raw_clean, raw_denoised, 
            output_prefix, metrics
        )
        
        # 2. 蝴蝶图对比
        self._plot_butterfly_comparison(
            raw_noisy, raw_clean, raw_denoised, output_prefix
        )
        
        # 3. 功率谱密度对比
        self._plot_psd_comparison(
            raw_noisy, raw_clean, raw_denoised, output_prefix
        )
        
        # 4. 拓扑图对比
        self._plot_topomap_comparison(
            raw_noisy, raw_clean, raw_denoised, output_prefix
        )
        
        # 5. 联合图
        self._plot_joint_comparison(
            raw_noisy, raw_clean, raw_denoised, output_prefix
        )
    
    def _plot_time_series_comparison(
        self,
        raw_noisy: mne.io.Raw,
        raw_clean: mne.io.Raw,
        raw_denoised: mne.io.Raw,
        output_prefix: str,
        metrics: Dict
    ):
        """绘制时间序列对比图"""
        # 选择前3个通道和前2秒数据
        channels = slice(0, 3)
        time_slice = slice(0, int(2 * raw_noisy.info['sfreq']))
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        time_axis = np.arange(time_slice.stop) / raw_noisy.info['sfreq']
        
        for ch_idx in range(3):
            # 噪声信号
            axes[ch_idx].plot(
                time_axis, 
                raw_noisy.get_data()[ch_idx, time_slice], 
                'r-', alpha=0.7, label='噪声信号', linewidth=1
            )
            
            # 干净信号
            axes[ch_idx].plot(
                time_axis, 
                raw_clean.get_data()[ch_idx, time_slice], 
                'g-', alpha=0.7, label='目标信号', linewidth=1
            )
            
            # 去噪信号
            axes[ch_idx].plot(
                time_axis, 
                raw_denoised.get_data()[ch_idx, time_slice], 
                'b-', alpha=0.7, label='去噪结果', linewidth=1
            )
            
            axes[ch_idx].set_ylabel(f'通道 {ch_idx+1}\n幅度 (fT)')
            axes[ch_idx].legend()
            axes[ch_idx].grid(True, alpha=0.3)
            
            if ch_idx == 0:
                # 在第一个子图添加指标信息
                metrics_text = (
                    f"SNR: {metrics.get('snr_db', 0):.1f} dB\n"
                    f"相关系数: {metrics.get('correlation', 0):.3f}\n"
                    f"SNR提升: {metrics.get('snr_improvement', 0):.1f} dB"
                )
                axes[ch_idx].text(
                    0.02, 0.98, metrics_text,
                    transform=axes[ch_idx].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                )
        
        axes[-1].set_xlabel('时间 (s)')
        plt.suptitle(f'MEG信号去噪对比 - {output_prefix}', fontsize=16)
        plt.tight_layout()
        
        output_path = self.output_dir / f'{output_prefix}_time_series.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"时间序列对比图已保存: {output_path}")
    
    def _plot_butterfly_comparison(
        self,
        raw_noisy: mne.io.Raw,
        raw_clean: mne.io.Raw,
        raw_denoised: mne.io.Raw,
        output_prefix: str
    ):
        """绘制蝴蝶图对比"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 限制时间范围以提高可视化效果
        tmax = min(2.0, raw_noisy.times[-1])
        
        # 原始噪声信号
        raw_noisy.plot(
            duration=tmax, n_channels=26, scalings='auto',
            show=False, title='原始噪声信号'
        )
        plt.savefig(self.output_dir / f'{output_prefix}_noisy_butterfly.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 目标干净信号
        raw_clean.plot(
            duration=tmax, n_channels=26, scalings='auto',
            show=False, title='目标干净信号'
        )
        plt.savefig(self.output_dir / f'{output_prefix}_clean_butterfly.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 去噪结果
        raw_denoised.plot(
            duration=tmax, n_channels=26, scalings='auto',
            show=False, title='去噪结果'
        )
        plt.savefig(self.output_dir / f'{output_prefix}_denoised_butterfly.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"蝴蝶图已保存: {output_prefix}_*_butterfly.png")
    
    def _plot_psd_comparison(
        self,
        raw_noisy: mne.io.Raw,
        raw_clean: mne.io.Raw,
        raw_denoised: mne.io.Raw,
        output_prefix: str
    ):
        """绘制功率谱密度对比"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        fmax = 100  # 最大频率
        
        # 计算PSD
        try:
            # 原始噪声信号
            raw_noisy.plot_psd(
                fmax=fmax, average=True, spatial_colors=False,
                show=False, ax=axes[0]
            )
            axes[0].set_title('原始噪声信号 PSD')
            
            # 目标干净信号
            raw_clean.plot_psd(
                fmax=fmax, average=True, spatial_colors=False,
                show=False, ax=axes[1]
            )
            axes[1].set_title('目标干净信号 PSD')
            
            # 去噪结果
            raw_denoised.plot_psd(
                fmax=fmax, average=True, spatial_colors=False,
                show=False, ax=axes[2]
            )
            axes[2].set_title('去噪结果 PSD')
            
            plt.tight_layout()
            
            output_path = self.output_dir / f'{output_prefix}_psd_comparison.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"PSD对比图已保存: {output_path}")
            
        except Exception as e:
            self.logger.warning(f"绘制PSD图失败: {e}")
            plt.close()
    
    def _plot_topomap_comparison(
        self,
        raw_noisy: mne.io.Raw,
        raw_clean: mne.io.Raw,
        raw_denoised: mne.io.Raw,
        output_prefix: str
    ):
        """绘制拓扑图对比"""
        try:
            # 计算每个通道的RMS值
            noisy_rms = np.sqrt(np.mean(raw_noisy.get_data() ** 2, axis=1))
            clean_rms = np.sqrt(np.mean(raw_clean.get_data() ** 2, axis=1))
            denoised_rms = np.sqrt(np.mean(raw_denoised.get_data() ** 2, axis=1))
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # 原始噪声信号拓扑图
            im1, _ = mne.viz.plot_topomap(
                noisy_rms, raw_noisy.info, axes=axes[0], 
                show=False, contours=0
            )
            axes[0].set_title('原始噪声信号 RMS')
            
            # 目标干净信号拓扑图
            im2, _ = mne.viz.plot_topomap(
                clean_rms, raw_clean.info, axes=axes[1], 
                show=False, contours=0
            )
            axes[1].set_title('目标干净信号 RMS')
            
            # 去噪结果拓扑图
            im3, _ = mne.viz.plot_topomap(
                denoised_rms, raw_denoised.info, axes=axes[2], 
                show=False, contours=0
            )
            axes[2].set_title('去噪结果 RMS')
            
            plt.tight_layout()
            
            output_path = self.output_dir / f'{output_prefix}_topomap_comparison.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"拓扑图对比已保存: {output_path}")
            
        except Exception as e:
            self.logger.warning(f"绘制拓扑图失败: {e}")
            plt.close()
    
    def _plot_joint_comparison(
        self,
        raw_noisy: mne.io.Raw,
        raw_clean: mne.io.Raw,
        raw_denoised: mne.io.Raw,
        output_prefix: str
    ):
        """绘制联合图"""
        try:
            # 选择时间点
            times = [0.5, 1.0, 1.5]  # 秒
            
            for i, signal_type in enumerate(['noisy', 'clean', 'denoised']):
                raw = [raw_noisy, raw_clean, raw_denoised][i]
                
                # 创建临时的evoked对象用于联合图
                data = raw.get_data()
                info = raw.info.copy()
                
                # 计算平均值
                evoked_data = np.mean(data.reshape(data.shape[0], -1, 1000), axis=1)
                
                evoked = mne.EvokedArray(evoked_data, info, tmin=0)
                
                # 绘制联合图
                fig = evoked.plot_joint(
                    times=times, title=f'{signal_type.capitalize()} Signal Joint Plot',
                    show=False
                )
                
                output_path = self.output_dir / f'{output_prefix}_{signal_type}_joint.png'
                fig.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            self.logger.info(f"联合图已保存: {output_prefix}_*_joint.png")
            
        except Exception as e:
            self.logger.warning(f"绘制联合图失败: {e}")
    
    def _generate_summary_report(self, all_metrics: List[Dict]):
        """生成汇总报告"""
        if not all_metrics:
            return
        
        # 计算统计指标
        metrics_summary = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m]
            if values:
                metrics_summary[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # 生成汇总图
        self._plot_metrics_summary(metrics_summary)
        
        # 保存汇总数据
        summary_file = self.output_dir / 'summary_metrics.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"汇总报告已保存: {summary_file}")
    
    def _plot_metrics_summary(self, metrics_summary: Dict):
        """绘制指标汇总图"""
        metrics_names = ['mse', 'mae', 'snr_db', 'correlation', 'snr_improvement']
        metrics_labels = ['MSE', 'MAE', 'SNR (dB)', 'Correlation', 'SNR Improvement (dB)']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (metric, label) in enumerate(zip(metrics_names, metrics_labels)):
            if metric in metrics_summary:
                data = metrics_summary[metric]
                
                # 创建误差条图
                axes[i].bar(
                    [label], [data['mean']], 
                    yerr=[data['std']], 
                    capsize=5, alpha=0.7
                )
                axes[i].set_ylabel(label)
                axes[i].set_title(f'{label}\n均值: {data["mean"]:.3f} ± {data["std"]:.3f}')
                axes[i].grid(True, alpha=0.3)
        
        # 删除多余的子图
        if len(metrics_names) < len(axes):
            for j in range(len(metrics_names), len(axes)):
                fig.delaxes(axes[j])
        
        plt.suptitle('MEG去噪性能汇总', fontsize=16)
        plt.tight_layout()
        
        output_path = self.output_dir / 'metrics_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"指标汇总图已保存: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='MEG去噪结果可视化')
    parser.add_argument('--model', type=str, required=True,
                       help='模型文件路径 (.pth)')
    parser.add_argument('--test_dir', type=str, default=r'F:\EX4\testdata_eptr2',
                       help='测试数据目录（绝对路径）')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    parser.add_argument('--sensor_info', type=str, 
                       default=r'E:\study\MEG\MEG\data\20230726zrcpos',
                       help='传感器位置信息文件路径')
    parser.add_argument('--output_dir', type=str, default='visualization_results',
                       help='输出目录')
    parser.add_argument('--max_files', type=int, default=10,
                       help='最大处理文件数')
    
    args = parser.parse_args()
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 创建可视化生成器
    visualizer = MEGVisualizationGenerator(
        model_path=args.model,
        config_path=args.config,
        sensor_info_path=args.sensor_info,
        output_dir=args.output_dir
    )
    
    # 处理数据集
    all_metrics = visualizer.process_dataset(
        test_dir=args.test_dir,
        max_files=args.max_files
    )
    
    print(f"\n可视化完成！共处理 {len(all_metrics)} 个文件")
    print(f"结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
