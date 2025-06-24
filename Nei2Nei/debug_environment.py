#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
环境检查和依赖验证脚本
运行此脚本确保所有依赖都正确安装
"""

import sys
import importlib
from pathlib import Path

def check_package(package_name, import_name=None):
    """检查包是否可以正常导入"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'Unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError as e:
        print(f"❌ {package_name}: 未安装或导入失败 - {e}")
        return False

def check_cuda():
    """检查CUDA可用性"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA: 可用 - 设备数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("⚠️  CUDA: 不可用，将使用CPU训练")
            return False
    except Exception as e:
        print(f"❌ CUDA检查失败: {e}")
        return False

def check_data_files():
    """检查数据文件"""
    print("\n=== 检查数据文件 ===")
    
    # 使用相对路径
    data_root = Path('./data')
    
    # 检查训练数据
    train_root = data_root / 'whq_train'
    train_noisy = train_root / 'noisy'
    train_clean = train_root / 'clean'
    
    # 检查验证数据
    valid_root = data_root / 'whq_valid'
    valid_noisy = valid_root / 'noisy'
    valid_clean = valid_root / 'clean'
    
    # 检查测试数据
    test_root = data_root / 'whq_test'
    test_noisy = test_root / 'noisy'
    test_clean = test_root / 'clean'
    
    # 检查目录是否存在
    dirs_to_check = [
        train_noisy, train_clean,
        valid_noisy, valid_clean,
        test_noisy, test_clean
    ]
    
    all_exists = True
    for dir_path in dirs_to_check:
        if not dir_path.exists():
            print(f"❌ 目录不存在: {dir_path}")
            all_exists = False
        else:
            print(f"✅ 目录存在: {dir_path}")
            # 检查.mat文件数量
            mat_files = list(dir_path.glob('*.mat'))
            print(f"   包含 {len(mat_files)} 个.mat文件")
    
    return all_exists

def main():
    print("=== MEG去噪项目环境检查 ===\n")
    
    # 检查Python版本
    python_version = sys.version
    print(f"Python版本: {python_version}\n")
    
    # 必需包列表
    required_packages = [
        ('torch', 'torch'),
        ('numpy', 'numpy'), 
        ('scipy', 'scipy'),
        ('matplotlib', 'matplotlib'),
        ('tqdm', 'tqdm'),
        ('pathlib', 'pathlib'),
    ]
    
    # 可选包列表
    optional_packages = [
        ('mne', 'mne'),
        ('torchvision', 'torchvision'),
        ('torchaudio', 'torchaudio'),
    ]
    
    print("检查必需依赖:")
    all_required_ok = True
    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            all_required_ok = False
    
    print("\n检查可选依赖:")
    for package_name, import_name in optional_packages:
        check_package(package_name, import_name)
    
    print("\n检查CUDA支持:")
    cuda_available = check_cuda()
    
    print("\n检查数据文件:")
    data_files_ok = check_data_files()
    
    print(f"\n=== 环境检查结果 ===")
    if all_required_ok:
        print("✅ 所有必需依赖都已正确安装!")
        if cuda_available:
            print("✅ 建议使用GPU训练以获得更好性能")
        else:
            print("⚠️  将使用CPU训练，速度可能较慢")
        if data_files_ok:
            print("✅ 数据文件已正确配置!")
        else:
            print("❌ 存在缺失的数据文件，请检查:")
        print("\n可以继续进行下一步调试!")
    else:
        print("❌ 存在缺失的依赖，请先安装:")
        print("pip install torch numpy scipy matplotlib tqdm")
        if not cuda_available:
            print("如需GPU支持，请安装CUDA版本的PyTorch")
    
    return all_required_ok

if __name__ == "__main__":
    main() 