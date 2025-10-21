#!/usr/bin/env python3
"""
消融实验脚本 - 逐步移除模型组件来验证每个组件的必要性
"""

import os
import json
import torch
import numpy as np
from sklearn.metrics import average_precision_score
import argparse
from transformers import AutoTokenizer, BertModel
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

# 导入项目模块
from src.dataset import DualStreamTCMDataset
from src.model import SyndromeClassifier
from src.utils import load_preprocessed_data_v13
from src.loss import HybridLoss

def collate_fn(batch):
    """自定义批处理函数"""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    
    cc_input_ids = torch.stack([item['cc_input_ids'] for item in batch])
    cc_attention_mask = torch.stack([item['cc_attention_mask'] for item in batch])
    od_input_ids = torch.stack([item['od_input_ids'] for item in batch])
    od_attention_mask = torch.stack([item['od_attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'cc_input_ids': cc_input_ids,
        'cc_attention_mask': cc_attention_mask,
        'od_input_ids': od_input_ids,
        'od_attention_mask': od_attention_mask,
        'labels': labels
    }

class AblationSyndromeClassifier(nn.Module):
    """消融实验版本的SyndromeClassifier"""
    
    def __init__(self, bert_path, knowledge_probes, ablation_type="full"):
        super(AblationSyndromeClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.hidden_size = self.bert.config.hidden_size
        self.ablation_type = ablation_type
        
        if ablation_type != "w/o_knowledge_probes":
            self.register_buffer('knowledge_probes', knowledge_probes)
        
        if ablation_type == "w/o_dual_stream":
            # 单流BERT，需要额外的分类头
            self.classifier = nn.Linear(self.hidden_size, knowledge_probes.shape[0])
    
    def forward(self, cc_input_ids, cc_attention_mask, od_input_ids, od_attention_mask):
        if self.ablation_type == "w/o_dual_stream":
            # 单流：拼接主诉和描述
            # 这里简化处理，实际应该更仔细地处理两个序列的拼接
            combined_input_ids = torch.cat([cc_input_ids, od_input_ids[:, 1:]], dim=1)  # 去掉第二个序列的[CLS]
            combined_attention_mask = torch.cat([cc_attention_mask, od_attention_mask[:, 1:]], dim=1)
            
            outputs = self.bert(input_ids=combined_input_ids, attention_mask=combined_attention_mask)
            pooled_output = outputs.pooler_output  # [batch, hidden]
            logits = self.classifier(pooled_output)  # [batch, num_labels]
            return logits
        
        else:
            # 双流处理
            cc_outputs = self.bert(input_ids=cc_input_ids, attention_mask=cc_attention_mask)
            od_outputs = self.bert(input_ids=od_input_ids, attention_mask=od_attention_mask)
            
            H_cc = cc_outputs.last_hidden_state  # [batch, seq_len_cc, hidden]
            H_od = od_outputs.last_hidden_state  # [batch, seq_len_od, hidden]
            
            if self.ablation_type == "w/o_knowledge_probes":
                # 不使用知识探针，直接使用BERT的pooled output
                cc_pooled = cc_outputs.pooler_output  # [batch, hidden]
                od_pooled = od_outputs.pooler_output  # [batch, hidden]
                
                if self.ablation_type == "w/o_coherence":
                    # 特征拼接（加法）
                    combined_features = cc_pooled + od_pooled
                    # 简单的线性分类器
                    logits = torch.matmul(combined_features, self.knowledge_probes.t())
                else:
                    # 证据一致性打分（乘法）
                    logits = (cc_pooled * od_pooled).sum(dim=-1, keepdim=True)
                    logits = logits.expand(-1, self.knowledge_probes.shape[0])
            else:
                # 使用知识探针
                E_k = self.knowledge_probes  # [num_labels, hidden]
                
                # 标签注意力机制
                C_l_cc = self.label_attention(E_k, H_cc, cc_attention_mask)
                C_l_od = self.label_attention(E_k, H_od, od_attention_mask)
                
                if self.ablation_type == "w/o_coherence":
                    # 特征拼接（加法）
                    combined_features = C_l_cc + C_l_od
                    logits = (combined_features * E_k.unsqueeze(0)).sum(dim=-1)
                else:
                    # 证据一致性打分（乘法）
                    logits = (C_l_cc * C_l_od).sum(dim=-1)
            
            return logits
    
    def label_attention(self, E_k, H, H_mask):
        """标签注意力机制"""
        attention_scores = torch.matmul(H, E_k.t()).permute(0, 2, 1)
        attention_scores = attention_scores / (self.hidden_size ** 0.5)
        attention_scores = attention_scores.masked_fill(H_mask.unsqueeze(1) == 0, -1e9)
        attention_probs = F.softmax(attention_scores, dim=-1)
        context_vector = torch.matmul(attention_probs, H)
        return context_vector

class AblationLoss(nn.Module):
    """消融实验版本的损失函数"""
    
    def __init__(self, ablation_type, distance_matrix=None, class_weights=None, device=None, 
                 lambda_focal=0.7, lambda_hls=0.3, gamma=2.0, temperature_hls=1.0):
        super(AblationLoss, self).__init__()
        self.ablation_type = ablation_type
        
        if ablation_type == "w/o_hybrid_loss":
            # 标准交叉熵损失
            self.loss_fn = nn.CrossEntropyLoss()
        elif ablation_type == "w/o_hls":
            # 只用Focal Loss
            self.focal_loss_fn = self.FocalLoss(alpha=class_weights.to(device), gamma=gamma)
        elif ablation_type == "w/o_focal":
            # 只用HLS Loss
            self.register_buffer("distance_matrix", distance_matrix.to(device))
            self.temperature_hls = temperature_hls
            self.num_classes = distance_matrix.shape[0]
        else:
            # 完整HybridLoss
            self.lambda_focal = lambda_focal
            self.lambda_hls = lambda_hls
            self.focal_loss_fn = self.FocalLoss(alpha=class_weights.to(device), gamma=gamma)
            self.register_buffer("distance_matrix", distance_matrix.to(device))
            self.temperature_hls = temperature_hls
            self.num_classes = distance_matrix.shape[0]
    
    def forward(self, logits, labels):
        if self.ablation_type == "w/o_hybrid_loss":
            return self.loss_fn(logits, labels)
        elif self.ablation_type == "w/o_hls":
            return self.focal_loss_fn(logits, labels)
        elif self.ablation_type == "w/o_focal":
            return self.hls_loss(logits, labels)
        else:
            # 完整HybridLoss
            focal_loss = self.focal_loss_fn(logits, labels)
            hls_loss = self.hls_loss(logits, labels)
            return self.lambda_focal * focal_loss + self.lambda_hls * hls_loss
    
    def FocalLoss(self, alpha, gamma):
        """Focal Loss实现"""
        def focal_loss_fn(logits, labels):
            ce_loss = F.cross_entropy(logits, labels, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = alpha[labels] * (1 - pt) ** gamma * ce_loss
            return focal_loss.mean()
        return focal_loss_fn
    
    def hls_loss(self, logits, labels):
        """HLS Loss实现"""
        probs = F.softmax(logits, dim=-1)
        
        # 创建目标分布
        target_dist = torch.zeros_like(probs)
        target_dist.scatter_(1, labels.unsqueeze(1), 1.0)
        
        # 计算层次距离权重
        batch_size = labels.size(0)
        distance_weights = torch.zeros(batch_size, self.num_classes, device=labels.device)
        
        for i in range(batch_size):
            true_label = labels[i].item()
            for j in range(self.num_classes):
                distance_weights[i, j] = self.distance_matrix[true_label, j]
        
        # 应用温度缩放
        distance_weights = distance_weights / self.temperature_hls
        
        # 计算加权KL散度
        kl_loss = F.kl_div(F.log_softmax(logits, dim=-1), target_dist, reduction='none')
        weighted_kl_loss = kl_loss * distance_weights
        return weighted_kl_loss.sum(dim=-1).mean()

def calculate_metrics(model, data_loader, device, num_classes, distance_matrix, dataset_name):
    """计算评估指标"""
    model.eval()
    all_probs = []
    all_labels = []
    all_predictions = []
    
    print(f"Evaluating {dataset_name} dataset...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch is None:
                continue
                
            cc_input_ids = batch['cc_input_ids'].to(device)
            cc_attention_mask = batch['cc_attention_mask'].to(device)
            od_input_ids = batch['od_input_ids'].to(device)
            od_attention_mask = batch['od_attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(cc_input_ids, cc_attention_mask, od_input_ids, od_attention_mask)
            probs = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predictions.cpu().numpy())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Processed {batch_idx + 1} batches...")
    
    # 合并结果
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    # 计算Macro-F1
    from sklearn.metrics import f1_score
    macro_f1 = f1_score(all_labels, all_predictions, average='macro')
    
    # 计算AUPRC
    auprc_scores = []
    for class_idx in range(num_classes):
        binary_labels = (all_labels == class_idx).astype(int)
        class_probs = all_probs[:, class_idx]
        if np.sum(binary_labels) > 0:
            auprc = average_precision_score(binary_labels, class_probs)
            auprc_scores.append(auprc)
        else:
            auprc_scores.append(0.0)
    macro_auprc = np.mean(auprc_scores)
    
    # 计算AHD
    distance_matrix_np = distance_matrix.cpu().numpy()
    ahd = 0.0
    for i in range(len(all_labels)):
        true_label = all_labels[i]
        pred_label = all_predictions[i]
        ahd += distance_matrix_np[true_label, pred_label]
    ahd /= len(all_labels)
    
    return {
        'macro_f1': macro_f1,
        'macro_auprc': macro_auprc,
        'ahd': ahd,
        'accuracy': np.mean(all_labels == all_predictions)
    }

def run_ablation_experiment(ablation_type, config, device):
    """运行单个消融实验"""
    print(f"\n{'='*60}")
    print(f"RUNNING ABLATION EXPERIMENT: {ablation_type}")
    print(f"{'='*60}")
    
    # 加载数据
    label2id, id2label, distance_matrix, class_weights, knowledge_probes = load_preprocessed_data_v13(config)
    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
    
    # 创建数据集
    test_dataset = DualStreamTCMDataset(config.test_path, tokenizer, label2id, config.max_len_cc, config.max_len_od)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    # 创建模型
    model = AblationSyndromeClassifier(config.bert_path, knowledge_probes.to(device), ablation_type)
    model.to(device)
    
    # 加载最优模型权重（如果存在）
    best_model_path = os.path.join(config.output_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        # 只加载兼容的权重
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded {len(pretrained_dict)} compatible parameters")
    
    # 计算指标
    metrics = calculate_metrics(model, test_loader, device, len(label2id), distance_matrix, "Test")
    
    print(f"\nResults for {ablation_type}:")
    print(f"  Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"  AUPRC: {metrics['macro_auprc']:.4f}")
    print(f"  AHD: {metrics['ahd']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_path", default="./ZY-BERT", help="BERT模型路径")
    parser.add_argument("--test_path", default="./data/test.json", help="测试数据路径")
    parser.add_argument("--preprocessed_data_dir", default="./preprocessed_data", help="预处理数据目录")
    parser.add_argument("--output_dir", default="./output_v13", help="输出目录")
    parser.add_argument("--max_len_cc", type=int, default=32, help="主诉最大长度")
    parser.add_argument("--max_len_od", type=int, default=256, help="其他描述最大长度")
    parser.add_argument("--batch_size", type=int, default=8, help="批大小")
    parser.add_argument("--no_cuda", action="store_true", help="不使用CUDA")
    
    args = parser.parse_args()
    
    # 添加缺失的路径参数
    args.label_map_path = os.path.join(args.preprocessed_data_dir, "label_map.json")
    args.distance_matrix_path = os.path.join(args.preprocessed_data_dir, "syndrome_distance_matrix")
    args.class_weights_path = os.path.join(args.preprocessed_data_dir, "class_weights.pt")
    args.knowledge_probes_path = os.path.join(args.preprocessed_data_dir, "knowledge_probes.pt")
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # 消融实验列表
    ablation_types = [
        "full",                    # Ours_Full (完整模型)
        "w/o_hls",                 # 移除HLS损失
        "w/o_focal",               # 移除Focal Loss
        "w/o_hybrid_loss",         # 移除HybridLoss，使用标准交叉熵
        "w/o_coherence",           # 移除证据一致性打分，使用特征拼接
        "w/o_dual_stream",         # 移除双流，使用单流BERT
    ]
    
    results = {}
    
    for ablation_type in ablation_types:
        try:
            metrics = run_ablation_experiment(ablation_type, args, device)
            results[ablation_type] = metrics
        except Exception as e:
            print(f"Error in {ablation_type}: {e}")
            results[ablation_type] = {"error": str(e)}
    
    # 保存结果
    results_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print("ABLATION STUDY RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for ablation_type, metrics in results.items():
        if "error" not in metrics:
            print(f"{ablation_type:20} | Macro-F1: {metrics['macro_f1']:.4f} | AUPRC: {metrics['macro_auprc']:.4f} | AHD: {metrics['ahd']:.4f}")
        else:
            print(f"{ablation_type:20} | ERROR: {metrics['error']}")
    
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()
