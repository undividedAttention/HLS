# 第3章 研究方法

本章详细阐述了我们提出的面向中医症候自动辨识的层次化标签相似性（Hierarchical Label Similarity, HLS）框架。该框架的核心思想是将中医辨证理论中固有的层次化知识，直接编码到模型的监督信号中，从而引导模型在预测时遵循临床逻辑，并有效缓解真实世界临床数据中普遍存在的类别不平衡与知识不完整问题。

## 3.1 问题形式化定义

我们将中医症候自动辨识任务形式化为一个层次化文本分类问题。

- **输入 (Input):** 对于每一个病历样本 $i$，其输入数据 $x_i$ 由多个文本字段构成，主要包括主诉（chief\_complaint）、现病史与四诊信息（description, detection）。我们将这些文本字段拼接为一个完整的文档 $x_i = (\text{chief\_complaint}_i; \text{description}_i; \text{detection}_i)$。

- **输出 (Output):** 模型的任务是预测该病历样本 $x_i$ 对应的最可能的症候标签 $y_i$。目标标签空间是一个包含 $M$ 个症候的集合 $C = \{c_1, c_2, \dots, c_M\}$。与传统分类任务不同，我们的模型将输出一个概率分布向量 $\mathbf{p}_i \in \mathbb{R}^M$，其中 $\mathbf{p}_i[j]$ 代表样本 $i$ 属于症候 $c_j$ 的概率。

- **外部知识 (External Knowledge):**
    1.  **层次结构知识 $T$:** 源于《中医病证分类与代码 GB/T 15657—2021》标准，通过处理`syndromeToPath.txt`文件构建成一个症候知识图谱（一个有向无环图，或森林）。该图谱 $T$ 定义了症候之间的父子、祖孙等“is-a”关系。
    2.  **文本定义知识 $K$:** 对于每个症候 $c_j \in C$，我们拥有其对应的外部知识文本 $k_j$，该文本由症候的“定义(Definition)”和“典型表现(Typical\_performance)”等描述性信息构成。

## 3.2 症候知识图谱构建与距离度量

为了让模型能够“理解”症候之间的亲疏远近关系，我们首先需要将`syndromeToPath.txt`中的文本路径，转化为一个可计算的图结构，并在此基础上定义一个有临床意义的距离度量。

### 3.2.1 知识图谱构建

我们使用`syndromeToPath.txt`文件中提供的146条推理路径来构建一个有向图 $T=(V, E)$，其中节点集合 $V$ 包含了所有的症候标签（叶子节点）以及它们的祖先节点（中间节点），边集合 $E$ 代表了节点间的父子关系。具体步骤如下：
1.  **节点与边的提取:** 逐行解析文件，对于每一条路径如“A -> B -> C”，我们提取所有节点{A, B, C}并建立有向边(A,B)和(B,C)。
2.  **虚拟根节点:** 为了将可能分离的辨证体系（形成森林）统一到一个单一的树结构中，我们引入一个全局的虚拟根节点`ROOT`。所有在路径中作为起点的节点（如上述的A）都将作为`ROOT`的直接子节点。
3.  **孤立节点处理:** 对于标准中未能找到路径的“风痰瘀阻证”和“正虚毒瘀证”等孤立节点，我们将其直接连接到虚拟根节点`ROOT`下，确保图中所有节点都是连通的。

### 3.2.2 层次距离度量

一个理想的距离度量应该能够量化将一个症候误诊为另一个的“严重程度”。基于构建的知识树 $T$，我们采用**最低公共祖先 (Lowest Common Ancestor, LCA)** 路径距离来定义任意两个症候 $c_i, c_j \in C$ 之间的诊断距离 $d(c_i, c_j)$：

$$d(c_i, c_j) = \text{depth}(c_i) + \text{depth}(c_j) - 2 \cdot \text{depth}(\text{LCA}(c_i, c_j))$$

其中，$\text{depth}(v)$ 表示节点 $v$ 在树 $T$ 中的深度（即从`ROOT`到该节点的距离）。此度量具有清晰的临床解释：LCA代表了两个症候在病理上共有的、最具体的上层概念。因此，该距离实际上衡量了从一个诊断分支“走”到另一个诊断分支所需的最短路径长度。例如，兄弟节点间的距离为2，而属于完全不同辨证体系的两个节点间的距离则会大得多。

## 3.3 模型架构

我们的模型由一个语义编码器和一个分类层组成，旨在将输入的病历文本映射到目标症候的概率分布上。

1.  **语义编码器 (Semantic Encoder):** 我们采用预训练的中文语言模型（如BERT）作为编码器骨干。输入的病历长文本 $x_i$ 首先被分词并输入BERT模型，得到一系列上下文感知的词向量表征。我们取`[CLS]`标记对应的输出向量作为整个病历文本的初步语义表征 $\mathbf{h}_i \in \mathbb{R}^H$，其中 $H$ 是BERT的隐藏层维度。

2.  **分类层 (Classification Layer):** 将初步语义表征 $\mathbf{h}_i$ 输入一个简单的全连接（Feed-Forward）网络，该网络将高维语义向量映射到大小为 $M$ 的logit向量 $\mathbf{z}_i \in \mathbb{R}^M$。

$$\mathbf{z}_i = \mathbf{W}\mathbf{h}_i + \mathbf{b}$$

其中 $\mathbf{W} \in \mathbb{R}^{M \times H}$ 和 $\mathbf{b} \in \mathbb{R}^M$ 是可学习的参数。最后，logit向量 $\mathbf{z}_i$ 经过`LogSoftmax`函数处理，得到最终的对数概率分布 $\log(\mathbf{p}_i)$。

## 3.4 层次化标签相似性 (HLS) 损失函数

这是我们框架的核心。传统的交叉熵损失将所有错误预测视为同等错误，这违背了临床逻辑。为此，我们设计了HLS损失函数，它通过优化模型预测分布与一个蕴含了层次知识的“软”目标分布之间的KL散度，来驱动模型学习。

### 3.4.1 层次化目标分布构建

对于一个真实的标签 $y_{\text{true}} \in C$，我们不再使用one-hot向量作为其目标，而是构建一个平滑的、层次感知的软目标分布 $\mathbf{y}'_{\text{soft}} \in \mathbb{R}^M$。其第 $j$ 个元素的值由症候 $c_j$ 与真实标签 $y_{\text{true}}$ 之间的层次距离 $d(y_{\text{true}}, c_j)$ 决定：

$$\mathbf{y}'_{\text{soft}}[j] = \frac{\exp(-d(y_{\text{true}}, c_j) / T)}{\sum_{k=1}^{M} \exp(-d(y_{\text{true}}, c_k) / T)}$$

其中，$T$ 是一个温度（temperature）超参数。当 $T \to \infty$ 时，分布趋向于均匀分布；当 $T \to 0$ 时，分布退化为one-hot分布。$T$ 控制了知识层次的“软化”程度。这个过程本质上是利用先验知识为每个样本“制造”了一个完美的教师概率分布。

### 3.4.2 HLS损失函数

我们的学习目标是最小化模型预测的概率分布 $\mathbf{p}_i$ 与我们构建的理想教师分布 $\mathbf{y}'_{\text{soft}}$ 之间的**Kullback-Leibler (KL) 散度**：

$$\mathcal{L}_{\text{HLS}}(y_i, \mathbf{p}_i) = D_{\text{KL}}(\mathbf{y}'_{\text{soft}} || \mathbf{p}_i) = \sum_{j=1}^{M} \mathbf{y}'_{\text{soft}}[j] \log\left(\frac{\mathbf{y}'_{\text{soft}}[j]}{\mathbf{p}_i[j]}\right)$$

由于模型输出的是对数概率 $\log(\mathbf{p}_i)$，实际计算时，损失函数更为便捷。最小化KL散度，会迫使模型的预测分布不仅要给正确答案高概率，还要使其整体形状逼近我们定义的、蕴含了完整层次结构信息的理想目标分布。这为模型提供了比单一正确答案丰富得多的监督信号，尤其有助于样本稀少类别的学习。

---

# 第3章 研究方法（当前实现，Dual-Stream + Label Attention + HybridLoss）

本节在延续前述形式化风格的基础上，严格对齐当前代码实现，给出最新版本的方法论描述。与初版相比，核心变化包括：
- 输入改为双通道文本编码（主诉/其他描述）；
- 标签以知识探针（由“定义+典型表现”经BERT编码得到的原型向量）进行语义初始化；
- 通过“标签注意力”从两路文本独立抽取针对每个标签的细粒度证据，再以证据一致性计算得到最终logits；
- 损失采用Focal Loss（类别不均衡）与HLS Loss（层次知识约束）的加权混合；
- 训练过程中每个epoch同时在dev与test评估，按dev Macro-F1保存最佳模型，并记录所有epoch指标。

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