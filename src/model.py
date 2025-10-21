import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class SyndromeClassifier(nn.Module):
    def __init__(self, bert_path, knowledge_probes):
        """
        V13 K-FEN 模型.
        不再需要 num_labels，因为标签数量可以从 knowledge_probes 推断出来。
        """
        super(SyndromeClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.hidden_size = self.bert.config.hidden_size
        
        # 将预计算的知识探针注册为固定的、不参与训练的缓冲区
        self.register_buffer("knowledge_probes", knowledge_probes)
        
    def forward(self, cc_input_ids, cc_attention_mask, od_input_ids, od_attention_mask):
        # 1. 双流编码，获取词元级表示
        cc_outputs = self.bert(input_ids=cc_input_ids, attention_mask=cc_attention_mask)
        od_outputs = self.bert(input_ids=od_input_ids, attention_mask=od_attention_mask)
        
        H_cc = cc_outputs.last_hidden_state  # [batch, cc_len, hidden]
        H_od = od_outputs.last_hidden_state  # [batch, od_len, hidden]

        # 2. 双向证据提取 (Bidirectional Evidence Extraction)
        # 使用固定的知识探针作为Query
        # E_k_norm = F.normalize(self.knowledge_probes, p=2, dim=1)
        E_k = self.knowledge_probes
        
        # a. 提取主诉中的细粒度证据 C_l_cc
        C_l_cc = self.label_attention(E_k, H_cc, cc_attention_mask) # [batch, num_labels, hidden]
        
        # b. 提取其他描述中的细粒度证据 C_l_od
        C_l_od = self.label_attention(E_k, H_od, od_attention_mask) # [batch, num_labels, hidden]
        
        # 3. 证据一致性评分 (Evidence Coherence Scoring)
        # 计算主观证据和客观证据之间的一致性（点积相似度）作为最终logits
        logits = (C_l_cc * C_l_od).sum(dim=-1) # [batch, num_labels]
        
        return logits

    def label_attention(self, E_k, H, H_mask):
        """
        通用的标签注意力模块
        E_k: 知识探针 (Query) [num_labels, hidden]
        H: 文本的词元级嵌入 (Key, Value) [batch, seq_len, hidden]
        H_mask: 文本的注意力掩码 [batch, seq_len]
        """
        # [batch, num_labels, seq_len]
        attention_scores = torch.matmul(H, E_k.t()).permute(0, 2, 1)
        attention_scores = attention_scores / (self.hidden_size ** 0.5)
        
        # 应用掩码，将padding位置的分数设为负无穷
        attention_scores = attention_scores.masked_fill(H_mask.unsqueeze(1) == 0, -1e9)
        
        # [batch, num_labels, seq_len]
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # [batch, num_labels, hidden]
        context_vector = torch.matmul(attention_probs, H)
        
        return context_vector

