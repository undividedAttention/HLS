# 消融实验结果（基于 dev_macro_f1 选择最佳 epoch）

## 最佳结果汇总

### 实验1 - Single-Stream + HybridLoss

**基于 dev_macro_f1 的最佳 epoch**: 12

**验证集结果**:
- Accuracy: 81.10%
- Macro-F1: **55.33%** ⭐
- AHD: 1.59
- AUPRC: 55.82%

**对应测试集结果**:
- Accuracy: **82.46%**
- Macro-F1: **56.52%**
- AHD: **1.46**
- AUPRC: 56.71%

---

### 实验2 - Dual-Stream + FocalOnly

**基于 dev_macro_f1 的最佳 epoch**: 13

**验证集结果**:
- Accuracy: 77.71%
- Macro-F1: **53.15%** ⭐
- AHD: 1.85
- AUPRC: 56.44%

**对应测试集结果**:
- Accuracy: **79.35%**
- Macro-F1: **53.97%**
- AHD: 1.73
- AUPRC: **59.30%**

---

## 三个实验配置回顾

### 1. 主实验（Baseline）- Dual-Stream + HybridLoss
**完整数据**: `output_v13/all_epochs_metrics.json`
- 架构: 双流BERT分别编码主诉和病史+四诊信息
- 机制: 双流注意力 + 逐元素点积
- 损失: 0.7 × FocalLoss + 0.3 × HLSLoss
- 注意: 只有1个epoch的数据

### 2. 实验1 - Single-Stream + HybridLoss  
**完整数据**: `output_ablation_exp1/all_epochs_metrics.json`
- 架构: 单流BERT（拼接输入）
- 机制: 直接与知识探针计算相似性
- 损失: 0.7 × FocalLoss + 0.3 × HLSLoss
- **消融内容**: 移除双流注意力机制和逐元素点积

### 3. 实验2 - Dual-Stream + FocalOnly
**完整数据**: `output_ablation_exp2/all_epochs_metrics.json`
- 架构: 与主实验相同（双流）
- 机制: 与主实验相同
- 损失: 仅FocalLoss（无HLS正则化）
- **消融内容**: 移除HLSLoss组件

---

## 关键发现

### 1. 单流 vs 双流架构
- ✅ 单流架构（实验1）在多个指标上表现更优
  - Test Accuracy: 82.46%（实验1）> 79.35%（实验2）
  - Test Macro-F1: 56.52%（实验1）> 53.97%（实验2）
  - Test AHD: 1.46（实验1）< 1.73（实验2）
- ⚠️ **结论**: 拼接输入也能达到更好的效果，双流设计的必要性受到挑战

### 2. HLS Loss 的效果
- ❌ 无HLS Loss的实验2表现较差
  - Test Accuracy: 79.35%（实验2）< 82.46%（实验1）
  - Test Macro-F1: 53.97%（实验2）< 56.52%（实验1）
  - Test AHD: 1.73（实验2）> 1.46（实验1）
- ⚠️ **结论**: HLS Loss在层次距离约束上有积极作用

### 3. AUPRC 表现
- 实验2在AUPRC上表现最好（59.30%）
- 但综合考虑其他指标，实验1整体更优

---

## 最终推荐

**综合评分（不考虑主实验，因其只有1个epoch）**:

| 实验 | Accuracy | Macro-F1 | AHD | AUPRC | 综合评分 |
|------|----------|----------|-----|-------|----------|
| 实验1 | 82.46% | 56.52% | 1.46 | 56.71% | ⭐⭐⭐⭐⭐ |
| 实验2 | 79.35% | 53.97% | 1.73 | 59.30% | ⭐⭐⭐⭐ |

**推荐配置**: **实验1（Single-Stream + HybridLoss）**

**理由**:
1. 准确率更高（82.46%）
2. Macro-F1更优（56.52%）
3. 层次距离最小（1.46）
4. AUPRC可接受（56.71%）
5. 架构更简洁（便于实现和维护）

**后续改进方向**:
1. 增加主实验的训练轮数（仅1个epoch数据不足）
2. 探索其他层次的损失函数设计
3. 尝试集成实验1和实验2的优点


