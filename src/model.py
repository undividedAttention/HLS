import torch
import torch.nn as nn
from transformers import BertModel

class SyndromeClassifier(nn.Module):
    def __init__(self, bert_path, num_classes, dropout_prob=0.1):
        super(SyndromeClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_path)
        self.bert_config = self.bert.config
        bert_hidden_size = self.bert_config.hidden_size
        
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(bert_hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

