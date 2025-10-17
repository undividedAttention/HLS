import json
import torch
from torch.utils.data import Dataset
from src.utils import get_synonym_mapping

class TCMDataset(Dataset):
    def __init__(self, file_path, tokenizer, label2id, max_seq_len):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_seq_len = max_seq_len
        self.data = self._load_data(file_path)

    def _load_data(self, file_path):
        """加载数据，并确保标签是标准形式"""
        processed_data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                
                text = f"主诉：{item.get('chief_complaint', '')}。病史：{item.get('description', '')}。四诊信息：{item.get('detection', '')}"
                
                # 直接使用您提供的、已经合并好的 'merge_syndrome' 字段
                canonical_label = item.get('merge_syndrome')
                
                if canonical_label in self.label2id:
                    label_id = self.label2id[canonical_label]
                    processed_data.append({'text': text, 'label': label_id})
        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'token_type_ids': inputs['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

