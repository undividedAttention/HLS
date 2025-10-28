# 消融实验设计文档

## 实验目标

验证K-FEN双流TCM证候分类系统的关键创新点：
1. **双流架构 + 逐元素点乘** 的必要性
2. **HLS正则化** (层次化标签相似性损失) 的作用

---

## 实验1：单流架构 vs 双流架构

### 实验设计

**基准模型（Baseline）**：
- 双流编码：主诉（chief_complaint）和其他描述（description + detection）分别编码
- 标签注意力机制：分别提取每个流的证据
- 证据一致性评分：使用逐元素点乘 `(C_l_cc * C_l_od).sum(-1)` 得到logits

**消融模型（Single Stream）**：
- 单流编码：将主诉、病史、四诊信息直接拼接 `[CLS] 主诉 病史 四诊信息 [SEP]`
- 标签注意力机制：直接从拼接文本提取每个标签的证据
- 相似性计算：使用点积相似性得到logits

### 理论假设

- **假设1**：双流架构允许模型分别建模主诉（主观证据）和其他描述（客观证据）
- **假设2**：逐元素点乘能够捕捉两种证据类型的一致性
- **预期结果**：双流架构表现更好，特别是AHD（平均层次距离）指标

### 实现细节

**数据集**：
- `SingleStreamTCMDataset`: 拼接文本，最大长度256
- `DualStreamTCMDataset`: 分别编码，主诉32，其他描述256

**模型**：
- `SingleStreamClassifier`: 单流BERT + 标签注意力 + 点积相似性
- `SyndromeClassifier`: 双流BERT + 标签注意力 + 逐元素点乘

**训练配置**：
- 损失函数：HybridLoss (Focal 0.7 + HLS 0.3)
- Batch size: 16
- Learning rate: 2e-5
- Epochs: 20
- 其他超参数与基准模型相同

### 评估指标

- **Macro-F1**: 主要性能指标
- **Accuracy**: 准确率
- **AHD**: 平均层次距离（层次化结构的重要性）
- **Loss**: 损失值

---

## 实验2：仅Focal Loss vs HybridLoss

### 实验设计

**基准模型（Baseline）**：
- 损失函数：`HybridLoss = 0.7 * FocalLoss + 0.3 * HLSLoss`
- Focal Loss：处理类别不平衡
- HLS Loss：正则化层结构，使模型预测接近标签层次结构

**消融模型（Focal Only）**：
- 损失函数：`FocalOnlyLoss` (仅使用Focal Loss)
- 不使用HLS Loss

### 理论假设

- **假设1**：HLS正则化能够引导模型学习标签的层次结构
- **假设2**：不使用HLS Loss会导致模型在层次距离（AHD）上的表现变差
- **预期结果**：HybridLoss在AHD上表现更好，F1可能略有下降但整体更稳定

### 实现细节

**数据集**：
- `DualStreamTCMDataset`: 与基准模型相同

**模型**：
- `SyndromeClassifier`: 双流架构（与基准模型相同）
- 仅改变损失函数

**训练配置**：
- 损失函数1：`HybridLoss(lambda_focal=0.7, lambda_hls=0.3)`
- 损失函数2：`FocalOnlyLoss` (lambda_focal=1.0)
- Batch size: 16
- Learning rate: 2e-5
- Epochs: 20
- 其他超参数相同

### 评估指标

- **Macro-F1**: 主要性能指标
- **Accuracy**: 准确率
- **AHD**: 平均层次距离（**关键指标**，验证HLS Loss的作用）
- **Loss**: 损失值

---

## 完整实验流程

### 步骤1：基准模型训练（需要已有结果）

```bash
# 如果已有output_v13的结果，直接使用
# 否则需要重新训练基准模型
cd /home/jingxiaozhu/HLS
CUDA_VISIBLE_DEVICES=2 python src/main.py \
    --epochs 20 --batch_size 16 \
    --output_dir output_baseline
```

### 步骤2：实验1 - 单流架构训练

```bash
cd /home/jingxiaozhu/HLS
CUDA_VISIBLE_DEVICES=2 python ablation_exp1_single_stream.py \
    --epochs 20 --batch_size 16 \
    --output_dir output_exp1_single_stream
```

### 步骤3：实验2 - 仅Focal Loss训练

```bash
cd /home/jingxiaozhu/HLS
CUDA_VISIBLE_DEVICES=2 python ablation_exp2_no_hls.py \
    --epochs 20 --batch_size 16 \
    --output_dir output_exp2_no_hls
```

### 步骤4：结果分析和报告生成

```bash
cd /home/jingxiaozhu/HLS
python compare_ablation_results.py
```

---

## 预期实验结果

### 假设结果矩阵

| 模型 | Macro-F1 | Accuracy | AHD | 实验目的 |
|------|----------|----------|-----|---------|
| **Baseline** (双流 + HybridLoss) | 0.5594 | 0.83 | 0.42 | 基准模型 |
| **Exp1** (单流 + HybridLoss) | ~0.45-0.50 | ~0.75 | ~0.60 | 验证双流架构 |
| **Exp2** (双流 + FocalOnly) | ~0.52-0.55 | ~0.80 | ~0.55 | 验证HLS Loss |

### 关键观察点

1. **实验1**：
   - F1下降：验证双流架构的重要性
   - AHD增加：验证逐元素点乘的一致性评分机制
   
2. **实验2**：
   - F1略微下降或持平：说明Focal Loss本身有效
   - **AHD显著增加**：验证HLS Loss在层次化结构学习中的作用
   - 预测的层次距离更远

---

## 实验执行时间估算

- **实验1 (单流)**: 约3-4小时（20 epochs）
- **实验2 (Focal Only)**: 约3-4小时（20 epochs）
- **总计**: 约6-8小时

---

## 注意事项

1. **数据一致性**：所有实验使用相同的train/dev/test划分
2. **种子固定**：建议固定随机种子以确保可重现性
3. **GPU资源**：建议使用同一个GPU并清空显存
4. **保存模型**：每个实验都会保存best_model.pt
5. **评估指标**：重点关注AHD指标来验证层次化结构学习



