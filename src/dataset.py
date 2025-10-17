import torch
from torch.utils.data import Dataset
import json

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

