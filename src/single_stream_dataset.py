import torch
from torch.utils.data import Dataset
import json

class SingleStreamTCMDataset(Dataset):
    """
    用于实验1的数据集：单流架构
    将主诉(chief_complaint)、病史(description)、四诊信息(detection)拼接成单个文本
    """
    def __init__(self, file_path, tokenizer, label2id, max_len=512):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len
        
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
        
        # 拼接主诉、病史、四诊信息
        concatenated_text = chief_complaint + " " + other_description
        
        label_str = item.get("merge_syndrome")
        
        if label_str not in self.label2id:
            return None

        label = self.label2id[label_str]

        # 对拼接后的文本进行编码
        encoding = self.tokenizer.encode_plus(
            concatenated_text, 
            add_special_tokens=True, 
            max_length=self.max_len,
            padding='max_length', 
            truncation=True, 
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'concatenated_input_ids': encoding['input_ids'].flatten(),
            'concatenated_attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }



