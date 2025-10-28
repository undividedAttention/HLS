import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class SingleStreamClassifier(nn.Module):
    """
    消融实验1的模型：单流架构
    将主诉(chief_complaint)、病史(description)、四诊信息(detection)直接拼接
    然后通过BERT编码，与知识探针计算相似性得到logits
    不使用双流注意力和逐元素点乘
    """
    def __init__(self, bert_path, knowledge_probes):
        super(SingleStreamClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.hidden_size = self.bert.config.hidden_size
        
        # 将预计算的知识探针注册为固定的、不参与训练的缓冲区
        self.register_buffer("knowledge_probes", knowledge_probes)
        
    def forward(self, concatenated_input_ids, concatenated_attention_mask):
        """
        参数:
            concatenated_input_ids: [batch, total_seq_len] 拼接后的输入
            concatenated_attention_mask: [batch, total_seq_len] 拼接后的mask
        
        流程:
            1. 使用BERT编码拼接后的文本
            2. 获取所有token的表示
            3. 与知识探针计算相似性，得到每个标签的得分
        """
        # 1. 使用BERT编码拼接后的文本
        outputs = self.bert(
            input_ids=concatenated_input_ids,
            attention_mask=concatenated_attention_mask
        )
        
        H = outputs.last_hidden_state  # [batch, total_len, hidden]
        
        # 2. 使用标签注意力提取每个标签的证据
        # 这里使用label_attention而不是简单的pooling，保持与原始模型类似的机制
        E_k = self.knowledge_probes  # [num_labels, hidden]
        
        # 计算每个token与每个知识探针的相似性
        # H: [batch, seq_len, hidden], E_k: [num_labels, hidden]
        # attention_scores: [batch, num_labels, seq_len]
        attention_scores = torch.matmul(H, E_k.t()).permute(0, 2, 1)
        attention_scores = attention_scores / (self.hidden_size ** 0.5)
        
        # 应用掩码
        attention_scores = attention_scores.masked_fill(
            concatenated_attention_mask.unsqueeze(1) == 0, -1e9
        )
        
        # 获取注意力权重
        attention_probs = F.softmax(attention_scores, dim=-1)  # [batch, num_labels, seq_len]
        
        # 3. 加权聚合得到每个标签的证据向量
        context_vector = torch.matmul(attention_probs, H)  # [batch, num_labels, hidden]
        
        # 4. 计算logits：每个标签的证据向量与知识探针的点积
        logits = (context_vector * E_k.unsqueeze(0)).sum(dim=-1)  # [batch, num_labels]
        
        return logits



