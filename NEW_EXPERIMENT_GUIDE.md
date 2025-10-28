# 新实验指南

## 实验设计概述

### Main Model（基准模型）
- **架构**: 单流BERT（拼接所有文本）
- **分类器**: 标签注意力（以知识探针为Query）
- **损失函数**: HybridLoss (0.7 × FocalLoss + 0.3 × HLSLoss)

### Ablation Experiments

1. **Ablation 1**: 验证HLS Loss的必要性
   - Main Model - HLS Loss = 仅Focal Loss

2. **Ablation 2**: 验证标签注意力的必要性
   - Main Model - 标签注意力 = CLS + 线性分类层

3. **Ablation 3**: 验证知识探针的必要性
   - Main Model，但知识探针随机初始化

4. **Ablation 4** (可选): 双流架构
   - 原来的Baseline（用于对比）

---

## 执行流程

### 步骤1: 网格搜索

```bash
# 激活conda环境
source /opt/miniconda/etc/profile.d/conda.sh
conda activate hls_env

# 运行网格搜索（约10-15小时，81个组合 × 15 epochs）
CUDA_VISIBLE_DEVICES=0 python grid_search.py \
    --batch_size 16 \
    --epochs 15 \
    --output_dir output_grid_search
```

### 步骤2: 根据最优参数训练Main Model

```bash
# 查看最优参数
cat output_grid_search/best_config.json

# 训练Main Model（使用最优参数）
CUDA_VISIBLE_DEVICES=0 python train_main_model.py \
    --epochs 15 \
    --batch_size 16 \
    --learning_rate <最优LR> \
    --lambda_focal <最优LF> \
    --lambda_hls <最优LH> \
    --gamma <最优Gamma> \
    --output_dir output_main_model
```

### 步骤3: 运行消融实验

```bash
# Ablation 1: 无HLS Loss
CUDA_VISIBLE_DEVICES=0 python train_ablation1_no_hls.py \
    --epochs 15 \
    --batch_size 16 \
    --learning_rate <最优LR> \
    --gamma <最优Gamma> \
    --output_dir output_ablation1

# Ablation 2: 无标签注意力
CUDA_VISIBLE_DEVICES=0 python train_ablation2_no_label_attn.py \
    --epochs 15 \
    --batch_size 16 \
    --learning_rate <最优LR> \
    --lambda_focal <最优LF> \
    --lambda_hls <最优LH> \
    --gamma <最优Gamma> \
    --output_dir output_ablation2

# Ablation 3: 随机知识探针
CUDA_VISIBLE_DEVICES=0 python train_ablation3_random_probes.py \
    --epochs 15 \
    --batch_size 16 \
    --learning_rate <最优LR> \
    --lambda_focal <最优LF> \
    --lambda_hls <最优LH> \
    --gamma <最优Gamma> \
    --output_dir output_ablation3
```

---

## 评价指标

每个实验都会记录以下指标（dev和test）：
- `accuracy`
- `macro_f1`
- `macro_precision`
- `macro_recall`
- `avg_hier_distance` (AHD)
- `auprc`

指标保存在 `output_*/all_epochs_metrics.json` 中。

---

## 文件说明

### 训练脚本
- `grid_search.py`: 网格搜索最优参数
- `train_main_model.py`: 训练主模型
- `train_ablation1_no_hls.py`: 消融实验1
- `train_ablation2_no_label_attn.py`: 消融实验2
- `train_ablation3_random_probes.py`: 消融实验3

### 模型文件
- `src/single_stream_model.py`: 单流架构模型
- `src/simple_classifier.py`: 简单分类器（用于Ablation 2）
- `src/single_stream_model_random_probes.py`: 随机探针模型（用于Ablation 3）
- `src/loss.py`: HybridLoss
- `src/focal_only_loss.py`: 仅Focal Loss

### 数据集文件
- `src/single_stream_dataset.py`: 单流数据集

---

## 预期结果

### Ablation 1
- **预期**: Macro-F1下降，AHD显著上升
- **证明**: HLS Loss解决了"临床盲视"问题

### Ablation 2
- **预期**: Macro-F1显著下降
- **证明**: 标签注意力解决了"信息压缩谬误"问题

### Ablation 3
- **预期**: Macro-F1下降
- **证明**: 外部知识（定义、典型表现）的注入是有益的

---

## 注意

1. **网格搜索耗时较长**（3×3×3×3=81个组合 × 15 epochs，预计10-15小时），建议在screen中运行
2. 所有实验使用**相同的学习率和gamma**（来自网格搜索最优配置）
3. 保存所有epoch的指标以便后续分析

