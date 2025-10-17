import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torch.utils.data import Dataset
import json

class SyndromeClassifier(nn.Module):
    def __init__(self, bert_path, num_labels):
        super(SyndromeClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.hidden_size = self.bert.config.hidden_size
        
        self.label_embeddings = nn.Embedding(num_labels, self.hidden_size)
        
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, cc_input_ids, cc_attention_mask, od_input_ids, od_attention_mask):
        cc_outputs = self.bert(input_ids=cc_input_ids, attention_mask=cc_attention_mask)
        od_outputs = self.bert(input_ids=od_input_ids, attention_mask=od_attention_mask)
        
        CLS_cc = cc_outputs.pooler_output
        H_od = od_outputs.last_hidden_state

        E_l = self.label_embeddings.weight
        
        # --- S_match ---
        CLS_cc_norm = F.normalize(CLS_cc, p=2, dim=1)
        E_l_norm = F.normalize(E_l, p=2, dim=1)
        S_match = torch.matmul(CLS_cc_norm, E_l_norm.t())
        
        # --- S_label ---
        attention_scores = torch.matmul(H_od, E_l_norm.t()).permute(0, 2, 1)
        attention_scores = attention_scores / (self.hidden_size ** 0.5)
        
        attention_scores = attention_scores.masked_fill(od_attention_mask.unsqueeze(1) == 0, -1e9)
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        C_l = torch.matmul(attention_probs, H_od)
        
        CLS_cc_expanded = CLS_cc.unsqueeze(1)
        S_label = (CLS_cc_expanded * C_l).sum(dim=-1)
        
        # --- Final Logits ---
        alpha_sigmoid = torch.sigmoid(self.alpha)
        logits = alpha_sigmoid * S_match + (1 - alpha_sigmoid) * S_label
        
        return logits


class DualStreamTCMDataset(Dataset):
    def __init__(self, file_path, tokenizer, label2id, max_len_cc=32, max_len_od=256):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len_cc = max_len_cc
        self.max_len_od = max_len_od
        
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        chief_complaint = item.get("chief_complaint", "")
        other_description = item.get("description", "") + " " + item.get("detection", "")
        label_str = item.get("merge_syndrome")
        
        if label_str not in self.label2id:
            # 遇到无效标签时，返回None，由collate_fn处理
            return None

        label = self.label2id[label_str]

        cc_encoding = self.tokenizer.encode_plus(
            chief_complaint, add_special_tokens=True, max_length=self.max_len_cc,
            padding='max_length', truncation=True, return_attention_mask=True,
            return_tensors='pt',
        )
        
        od_encoding = self.tokenizer.encode_plus(
            other_description, add_special_tokens=True, max_length=self.max_len_od,
            padding='max_length', truncation=True, return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'cc_input_ids': cc_encoding['input_ids'].flatten(),
            'cc_attention_mask': cc_encoding['attention_mask'].flatten(),
            'od_input_ids': od_encoding['input_ids'].flatten(),
            'od_attention_mask': od_encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

