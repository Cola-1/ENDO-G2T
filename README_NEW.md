# ST-Endo4DGS with Geometric Priors 🏥✨

## 🎯 概述

ST-Endo4DGS 是一个专门针对内窥镜手术场景的4D高斯喷射重建方法，集成了先进的几何先验技术，用于提升动态内窥镜视频的新视角合成质量。

### 🔥 核心特性

- **🧠 智能几何先验**: 集成StreamVGGT深度估计先验，显著提升重建质量
- **⚡ 高效4D重建**: 基于4D高斯喷射的快速动态场景重建
- **🎯 内窥镜优化**: 专门针对内窥镜场景的光照和变形特点优化
- **📊 渐进式融合**: 智能的先验权重调度，避免训练不稳定
- **🔧 灵活配置**: 支持有/无几何先验的对比训练

## 🛠️ 系统架构

```
ST-Endo4DGS Pipeline with Geometric Priors
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Video   │───▶│  StreamVGGT      │───▶│  Depth Priors   │
│   Sequences     │    │  Depth Network   │    │  Generation     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                                               │
         ▼                                               ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Camera Poses  │───▶│   4D Gaussian    │◀───│  Prior-Guided   │
│   Estimation    │    │   Splatting      │    │  Optimization   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Novel View     │
                       │  Synthesis      │
                       └─────────────────┘
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 克隆项目
git clone https://github.com/your-repo/ST-Endo4DGS
cd ST-Endo4DGS

# 创建环境
conda env create --file environment.yml
conda activate st-endo4dgs

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

**内窥镜数据集结构:**
```
data/endonerf/pulling_soft_tissues/
├── images/                    # 原始图像序列
├── poses_bounds.npy          # 相机位姿
├── depths/                   # (可选) 深度图
└── priors/
    └── streamvggt/          # StreamVGGT几何先验
        ├── depth_*.npy      # 深度先验
        ├── normal_*.npy     # 法向量先验
        └── confidence_*.npy # 置信度掩码
```

**预处理数据:**
```bash
# 生成基础数据
python scripts/pre_dam_dep.py --dataset_root data/endonerf/pulling_soft_tissues --rgb_paths images

# 导出先验
python /root/autodl-tmp/ST-Endo4DGS-main/tools/vggt_export.py \
  --data_root /root/autodl-tmp/ST-Endo4DGS-main/data/endonerf/pulling_soft_tissues


# 生成几何先验 (如果需要)
# python tools/generate_priors.py --data_path data/endonerf/pulling_soft_tissues
```

## 🎯 训练指南

### 带几何先验训练 (推荐) ✅

```bash
# 使用几何先验的完整训练
python train1.py --config configs/endoNerf/pulling.yaml

# 自定义参数训练
python train1.py \
  --config configs/endoNerf/pulling.yaml \
  --iterations 7000 \
  --eval_interval 500 \
  --lambda_si 0.001 \
  --lambda_depth_grad 0.0003
```

**关键配置参数:**
- `use_vggt_priors: True` - 启用几何先验
- `lambda_si: 0.001` - Scale-Invariant损失权重
- `lambda_depth_grad: 0.0003` - 深度梯度损失权重
- `prior_warmup_steps: 1000` - 先验预热步数

### 无几何先验训练 (对比基线) ❌

```bash
# 纯4DGS基线训练
python train1.py --config configs/endoNerf/pulling_no_priors.yaml
```

### 训练监控

```bash
# 实时监控训练进度
python monitor_training.py --output_dir output/endonerf/pulling

# TensorBoard可视化
tensorboard --logdir output/endonerf/pulling/tb_logs
```

## 📊 几何先验技术详解

### 🧠 StreamVGGT深度先验

StreamVGGT是一个专门训练的深度估计网络，为内窥镜场景提供高质量的几何先验：

```python
# 先验集成示例
if use_vggt_priors:
    # 加载深度先验
    depth_prior = load_depth_prior(frame_idx)
    confidence_mask = load_confidence_mask(frame_idx)
    
    # Scale-Invariant损失
    si_loss = lambda_si * scale_invariant_loss(pred_depth, depth_prior, confidence_mask)
    
    # 深度梯度损失
    grad_loss = lambda_depth_grad * depth_gradient_loss(pred_depth, depth_prior)
    
    total_loss += si_loss + grad_loss
```

### ⚖️ 损失函数设计

**完整损失函数:**
```
L_total = L_color + λ_dssim × L_dssim + λ_depth × L_depth 
        + λ_si × L_si + λ_grad × L_depth_grad + λ_entropy × L_entropy
```

**各项说明:**
- `L_color`: RGB重建损失 (L1)
- `L_dssim`: 结构相似性损失
- `L_depth`: 深度重建损失
- `L_si`: Scale-Invariant深度损失 ⭐
- `L_depth_grad`: 深度梯度损失 ⭐
- `L_entropy`: 不透明度熵正则化

### 📈 渐进式先验融合

为避免训练不稳定，采用渐进式先验权重调度：

```python
def get_prior_weight(iteration, warmup_steps=1000, max_weight=0.01):
    if iteration < warmup_steps:
        return 0.0
    else:
        progress = min(1.0, (iteration - warmup_steps) / warmup_steps)
        return max_weight * progress
```

## 🎮 渲染与评估

### 高质量渲染

```bash
# 渲染测试集
python render.py \
  --config configs/endoNerf/pulling.yaml \
  --checkpoint output/endonerf/pulling/chkpnt_best.pth \
  --skip_train --skip_video  

# 高帧率性能测试
python render.py \
  --config configs/endoNerf/pulling.yaml \
  --checkpoint output/endonerf/pulling/chkpnt_best.pth \
  --skip_train --skip_video \
  --measure_raster_only
```

### 定量评估

```bash
# 计算所有指标
python metrics.py -m output/endonerf/pulling

# 生成评估报告
python tools/generate_report.py --output_dir output/endonerf/pulling
```



## 🔧 高级配置

### 配置文件模板

```yaml
# configs/custom_config.yaml
gaussian_dim: 4
time_duration: [0.0, 1.0]
num_pts: 300_000
batch_size: 8

ModelParams:
  sh_degree: 3
  source_path: "your/data/path"
  model_path: "your/output/path"
  # 几何先验设置
  use_vggt_priors: True
  vggt_prior_dir: "your/priors/path"

OptimizationParams:
  iterations: 7000
  # 先验权重配置
  use_scale_depth: True
  lambda_si: 0.001
  lambda_depth_grad: 0.0003
  prior_warmup_steps: 1000
  prior_max_weight: 0.01
```

### 自定义先验

```python
# 添加自定义几何先验
def custom_prior_loss(gaussians, camera, gt_image):
    # 实现您的先验损失
    custom_loss = your_prior_function(gaussians, camera)
    return custom_loss
```

## 🐛 故障排除

### 常见问题

**Q1: 几何先验加载失败**
```bash
# 检查先验数据完整性
python tools/validate_priors.py --prior_dir data/endonerf/pulling_soft_tissues/priors/streamvggt
```

**Q2: 训练内存不足**
```yaml
# 降低批大小和点数
batch_size: 4
num_pts: 200_000
```

**Q3: 收敛速度慢**
```yaml
# 调整学习率和先验权重
position_lr_init: 0.0002
lambda_si: 0.0005
prior_warmup_steps: 500
```

## 🔬 技术细节

### 关键创新点

1. **几何感知的4D高斯**: 结合深度先验的高斯优化
2. **渐进式先验融合**: 避免训练早期的先验冲突
3. **内窥镜特化损失**: 针对内窥镜场景的损失设计
4. **多尺度深度监督**: 不同分辨率的深度一致性约束

### 代码架构

```
ST-Endo4DGS/
├── train1.py              # 主训练脚本
├── gaussian_renderer/     # 渲染核心
├── scene/                 # 场景管理
├── utils/                 # 工具函数
│   ├── loss_utils.py     # 损失函数
│   └── prior_utils.py    # 先验处理
├── configs/              # 配置文件
└── tools/                # 辅助工具
```

