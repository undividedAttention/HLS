"""
Single Stream Model with Random Probes
用于Ablation 3：验证知识探针的必要性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class SingleStreamClassifierRandomProbes(nn.Module):
    """单流架构，但使用随机初始化的知识探针"""
    
    def __init__(self, bert_path, random_probes):
        """
        Args:
            bert_path: BERT模型路径
            random_probes: 随机初始化的探针 [num_labels, hidden_size]
        """
        super(SingleStreamClassifierRandomProbes, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.hidden_size = self.bert.config.hidden_size
        
        # 使用随机初始化的探针（可训练）
        self.register_parameter("knowledge_probes", nn.Parameter(random_probes))
        
        # 标签注意力层
        self.label_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=0.1
        )
    
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
        
        Returns:
            logits: [batch, num_labels]
        """
        # 1. BERT编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        H = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        
        # 2. 标签注意力
        # knowledge_probes: [num_labels, hidden_size]
        E_k = self.knowledge_probes.unsqueeze(0)  # [1, num_labels, hidden_size]
        E_k = E_k.expand(H.size(0), -1, -1)  # [batch, num_labels, hidden_size]
        
        # Query: E_k, Key/Value: H
        E_k_t = E_k.transpose(0, 1)  # [num_labels, batch, hidden_size]
        H_t = H.transpose(0, 1)  # [seq_len, batch, hidden_size]
        
        C_l, _ = self.label_attention(E_k_t, H_t, H_t, key_padding_mask=~attention_mask.bool())
        # C_l: [num_labels, batch, hidden_size]
        C_l = C_l.transpose(0, 1)  # [batch, num_labels, hidden_size]
        
        # 3. 计算每个标签的相似性分数
        # 使用探针和提取的证据之间的点积
        logits = (C_l * E_k).sum(dim=-1)  # [batch, num_labels]
        
        return logits


def create_random_probes_model(bert_path, num_labels, hidden_size=768):
    """创建使用随机探针的模型"""
    random_probes = torch.randn(num_labels, hidden_size)
    return SingleStreamClassifierRandomProbes(bert_path, random_probes)

