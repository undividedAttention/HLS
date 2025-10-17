import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

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

