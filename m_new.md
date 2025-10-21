# 第3章 研究方法（当前实现，Dual-Stream + Label Attention + HybridLoss）

## 3.1 问题形式化

- **输入 (Dual-Stream Input):** 对于样本 \(i\)，我们显式拆分文本为两路：主诉文本 \(x^{(cc)}_i\) 与其他描述文本 \(x^{(od)}_i\)，其中其他描述由现病史与四诊信息拼接而成。两路文本分别经过同一个中文BERT编码器处理。
- **输出 (Output):** 模型输出对所有症候标签集合 \(C=\{c_1,\dots,c_M\}\) 的打分向量 \(\mathbf{z}_i\in\mathbb{R}^M\)，其经Softmax得到概率分布 \(\mathbf{p}_i\)。
- **外部知识 (External Knowledge):**
  - 层次结构知识 \(T\)：由标准路径文件构建，提供层次距离 \(d(c_i,c_j)\)。
  - 文本定义知识 \(K\)：每个标签 \(c_j\) 的“Definition”与“Typical_performance”，用于构建标签原型（知识探针）。

## 3.2 预处理与先验构建

1. **标签映射与距离矩阵：** 由`label_map.json`获得 \(\text{id2label},\text{label2id}\)，并基于知识图谱 \(T\) 计算 \(M\times M\) 距离矩阵 \(D\)，元素 \(D_{ij}=d(c_i,c_j)\)。
2. **类别权重 \(\alpha\)（不均衡校正）：** 读取训练集`train.json`（JSONL），统计各标签样本数 \(n_j\)，采用“有效样本数”法（Effective Number of Samples）计算权重
   \[ w_j = \frac{1-\beta}{1-\beta^{n_j}} \,,\quad \alpha = \text{normalize}(\mathbf{w}) \,. \]
   权重以Tensor形式保存为`class_weights.pt`。
3. **知识探针（标签原型向量） \(E\)：** 对每个标签 \(c_j\)，将其“Definition”与“Typical\_performance”拼接为语义文本；使用与模型相同的BERT将该文本编码，并取`pooler_output`作为原型向量 \(\mathbf{e}_j\)。所有标签原型堆叠为矩阵 \(E\in\mathbb{R}^{M\times H}\)，保存为`knowledge_probes.pt`。这里 \(H\) 为BERT隐藏维度，当前实现对齐`ZY-BERT`为1024。

## 3.3 模型架构（SyndromeClassifier）

1. **双通道BERT编码：**
   - 主诉：\( H^{(cc)} \in \mathbb{R}^{B\times L_{cc}\times H} \leftarrow \text{BERT}(x^{(cc)}) \)
   - 其他描述：\( H^{(od)} \in \mathbb{R}^{B\times L_{od}\times H} \leftarrow \text{BERT}(x^{(od)}) \)
   其中 \(B\) 为batch，\(L_{cc},L_{od}\) 为两路序列长度。

2. **标签注意力（Label Attention）：** 给定标签原型矩阵 \(E\in\mathbb{R}^{M\times H}\)，对每一路隐藏状态 \(H\in\mathbb{R}^{B\times L\times H}\) 计算：
   - 分数：\( A = \text{softmax}\Big( \frac{H E^\top}{\sqrt{H}} \Big) \in \mathbb{R}^{B\times L\times M} \)
   - 证据：\( C = A^\top H \in \mathbb{R}^{B\times M\times H} \)
   实现中等价为对每个标签原型作为Query，与词元表征进行加权汇聚，得到“针对每个标签的文本证据”。分别记 \(C^{(cc)}\) 与 \(C^{(od)}\)。

3. **证据一致性打分（Evidence Coherence Scoring）：**
   对同一标签的两路证据向量做元素乘并在隐藏维上求和：
   \[ z_{i,j} = \langle C^{(cc)}_{i,j,:},\; C^{(od)}_{i,j,:} \rangle = \sum_{h=1}^{H} C^{(cc)}_{i,j,h} \cdot C^{(od)}_{i,j,h} \,, \quad \mathbf{z}_i=(z_{i,1},\dots,z_{i,M}). \]
   该打分鼓励“主诉证据”与“其他描述证据”在正确标签上具有一致性。

## 3.4 损失函数（HybridLoss）

训练目标由两部分组成：

1. **Focal Loss（类别不均衡鲁棒）：**
   \[ \mathcal{L}_{\text{Focal}} = -\sum_{i} \alpha_{y_i}\; (1-p_{i,y_i})^\gamma\; \log p_{i,y_i} \,, \]
   其中 \(\alpha\) 来自`class_weights.pt`，\(\gamma\) 为聚焦参数，抑制易分类样本的主导。

2. **HLS Loss（层次结构对齐）：**
   给定真实标签 \(y_i\)，基于距离矩阵 \(D\) 与温度 \(T\) 构造软目标分布：
   \[ y'_{i,j} = \frac{\exp(-D_{y_i,j}/T)}{\sum_k \exp(-D_{y_i,k}/T)} \,, \quad \mathcal{L}_{\text{HLS}} = D_{\text{KL}}(\mathbf{y}'_i\;\Vert\;\mathbf{p}_i). \]

3. **加权混合目标：**
   \[ \mathcal{L} = \lambda_{\text{focal}}\, \mathcal{L}_{\text{Focal}} + \lambda_{\text{hls}}\, \mathcal{L}_{\text{HLS}}. \]

## 3.5 训练与评估协议

- **优化与调度：** 采用AdamW优化器与线性warmup余弦/线性衰减调度（实现为线性warmup+线性衰减）。
- **多卡：** 若存在多GPU，使用`nn.DataParallel`。
- **批量与长度：** 主诉最大长度`max_len_cc`，其他描述`max_len_od`，批量`batch_size`可配置。
- **每epoch评估：**
  - 在dev与test两套数据上均计算：Loss、Accuracy、Macro-F1、平均层次距离（AHD）。
  - 将所有epoch在两数据集上的指标累计写入`output_v13/all_epochs_metrics.json`。
  - 以dev Macro-F1为准保存`best_model.pt`到`output_v13/`。
- **资产再生成：** 提供`--regen_assets`选项，可在运行前基于当前`train.json`与`syndrome_knowledge.json`重新计算并保存`class_weights.pt`与`knowledge_probes.pt`。

## 3.6 数据与实现细节

- **数据格式：** 训练/验证/测试均为JSONL；每行包含`chief_complaint`、`description`、`detection`与标准化标签`merge_syndrome`。
- **数据集构建：** `DualStreamTCMDataset` 分别对两路文本进行分词编码，返回`cc_input_ids/attention_mask`与`od_input_ids/attention_mask`以及`labels`。
- **标签原型维度一致性：** `knowledge_probes.pt`维度与`ZY-BERT`隐藏维度一致（当前为1024），避免维度不匹配。
- **度量实现：**
  - Accuracy与Macro-F1按常规定义计算；
  - AHD基于预测与真实标签之间的层次距离矩阵 \(D\) 求样本平均。

## 3.7 方法学优势与讨论

- **显式利用结构化先验：** 通过HLS Loss将层次知识融入监督信号，使“近错轻、远错重”，与临床语义一致。
- **跨模态（主观/客观）一致性：** 双通道与证据一致性打分，促使模型在主诉与其他描述间学习到一致的标签证据。
- **小样本类别更友好：** Focal Loss结合有效样本数权重，显著缓解类别不均衡问题。
- **可解释性增强：** 标签注意力为每个标签产出对应的文本证据向量，便于溯源与分析。

以上描述完整对应当前代码实现（`src/model.py`, `src/loss.py`, `src/dataset.py`, `src/utils.py`, `src/main.py`）中的架构、目标与训练评估流程。