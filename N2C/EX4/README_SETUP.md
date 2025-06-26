# MEG信号去噪项目设置指南

## 项目结构
```
EX4/
├── config.json                    # 主配置文件
├── visualize_results.py          # 主运行脚本（包含去噪和可视化功能）
├── train.py                      # 模型训练脚本
├── dataset.py                    # 数据加载模块
├── conformer_denoiser.py         # 模型定义
├── loss.py                       # 损失函数定义
├── traindata_eptr1/              # 训练数据目录
│   ├── clean/                    # 干净信号
│   └── noisy/                    # 噪声信号
├── testdata_eptr2/               # 测试数据目录
│   ├── clean/                    # 干净信号
│   └── noisy/                    # 噪声信号
├── runs/                         # 模型输出目录
│   └── meg_conformer_train/      # 训练好的模型
└── visualization_results/        # 可视化结果输出
    ├── denoised_data/           # 去噪后的数据
    └── concatenated_results/    # 拼接和可视化结果
```

## 环境要求
- Python 3.8+
- PyTorch
- MNE-Python
- NumPy
- SciPy
- Matplotlib
- tqdm
- scikit-learn

## 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 生成数据（可选）
如果需要重新生成训练和测试数据：
```bash
# 生成训练数据（保存到 traindata_eptr1/ 目录）
python traindata_generate(eptr1).py

# 生成测试数据（保存到 testdata_eptr2/ 目录）
python testdata_generate(eptr2).py
```

### 2. 训练模型
```bash
python train.py --config config.json
```

### 3. 运行去噪和可视化（主要使用方式）
```bash
# 使用默认配置
python visualize_results.py

# 指定自定义参数
python visualize_results.py --model runs/meg_conformer_train/latest_checkpoint.pth --test_dir testdata_eptr2
```

### 4. 命令行参数说明
- `--model`: 模型权重文件路径（默认：runs/meg_conformer_train/latest_checkpoint.pth）
- `--test_dir`: 测试数据目录（默认：testdata_eptr2）
- `--denoised_dir`: 去噪数据保存目录（默认：visualization_results/denoised_data）
- `--output_dir`: 可视化结果输出目录（默认：visualization_results/concatenated_results）
- `--run_denoise`: 是否运行去噪过程（默认启用）
- `--force_run_denoise`: 强制运行去噪过程，即使已存在去噪数据

## 输出说明

### 去噪数据
- 保存在 `visualization_results/denoised_data/` 目录
- 文件格式：`denoised_X.mat`，其中X为样本编号
- 数据格式：MAT文件，包含键值'data'，形状为(channels, time)

### 可视化结果
保存在 `visualization_results/concatenated_results/` 目录：
- `concatenated_denoised.fif`: MNE格式的拼接数据
- `concatenated_denoised.mat`: MAT格式的拼接数据
- `concatenated_psd.png`: 功率谱密度图
- `concatenated_evoked.png`: Evoked响应图
- `concatenated_topomap.png`: 拓扑图
- `concatenated_butterfly.png`: Butterfly图
- `concatenated_joint.png`: Joint图

### 评估结果
- `evaluation_results.json`: 包含MSE、MAE、SNR、相关系数等评估指标

## 配置文件说明

### config.json
主配置文件，包含以下部分：
- `data`: 数据相关配置（目录路径、预处理参数等）
- `model`: 模型架构参数
- `loss`: 损失函数配置
- `training`: 训练参数
- `output_dir`: 模型输出目录

## 数据格式要求
- 训练和测试数据应放在相应目录的`clean/`和`noisy/`子文件夹中
- 数据文件格式：`.mat`文件
- 文件命名：`clean_X.mat`和`noisy_X.mat`，其中X为样本编号
- 数据结构：MAT文件应包含键值'data'，数据形状为(channels, time)或(time, channels)

## 移植说明
该项目使用相对路径，可以整体复制到任何位置运行。只需确保：
1. 保持目录结构不变
2. 安装所需依赖
3. 根据需要修改config.json中的参数

## 故障排除
1. **找不到模型文件**：检查`runs/meg_conformer_train/`目录是否存在模型权重文件
2. **找不到数据文件**：确认测试数据目录结构正确，包含`clean/`和`noisy/`子文件夹
3. **内存不足**：可以在config.json中减小batch_size参数
4. **CUDA错误**：如果没有GPU，程序会自动使用CPU，但速度较慢 