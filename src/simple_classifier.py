"""
Simple Classifier: 全局池化 + 线性分类层
替代标签注意力机制，用于Ablation 2
"""

import torch
import torch.nn as nn
from transformers import BertModel, AutoTokenizer


class SimpleClassifier(nn.Module):
    """简单的分类器：CLS标记 + 全连接层"""
    
    def __init__(self, bert_path, num_labels):
        super(SimpleClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.hidden_size = self.bert.config.hidden_size
        self.num_labels = num_labels
        
        # 简单的线性分类层
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
        
        Returns:
            logits: [batch, num_labels]
        """
        # BERT编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 使用CLS标记
        cls_output = outputs.last_hidden_state[:, 0]  # [batch, hidden_size]
        
        # 分类
        logits = self.classifier(cls_output)  # [batch, num_labels]
        
        return logits


def create_simple_classifier(bert_path, num_labels):
    """创建简单的分类器"""
    return SimpleClassifier(bert_path, num_labels)

