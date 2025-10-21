#!/usr/bin/env python3
"""
修正版消融实验 - 区分训练时消融和推理时消融
"""

import os
import json
import torch
import numpy as np
from sklearn.metrics import average_precision_score, f1_score
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

class InferenceAblationSyndromeClassifier(nn.Module):
    """推理时消融的SyndromeClassifier"""
    
    def __init__(self, bert_path, knowledge_probes, ablation_type="full"):
        super(InferenceAblationSyndromeClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        self.hidden_size = self.bert.config.hidden_size
        self.ablation_type = ablation_type
        
        if ablation_type != "w/o_knowledge_probes":
            self.register_buffer('knowledge_probes', knowledge_probes)
    
    def forward(self, cc_input_ids, cc_attention_mask, od_input_ids, od_attention_mask):
        if self.ablation_type == "w/o_dual_stream":
            # 单流：拼接主诉和描述
            # 更仔细的拼接处理
            batch_size = cc_input_ids.size(0)
            max_len = cc_input_ids.size(1) + od_input_ids.size(1) - 1  # 减去重复的[CLS]
            
            # 创建新的input_ids和attention_mask
            combined_input_ids = torch.zeros(batch_size, max_len, dtype=cc_input_ids.dtype, device=cc_input_ids.device)
            combined_attention_mask = torch.zeros(batch_size, max_len, dtype=cc_attention_mask.dtype, device=cc_attention_mask.device)
            
            # 拼接序列
            combined_input_ids[:, :cc_input_ids.size(1)] = cc_input_ids
            combined_input_ids[:, cc_input_ids.size(1):] = od_input_ids[:, 1:]  # 去掉[CLS]
            
            combined_attention_mask[:, :cc_attention_mask.size(1)] = cc_attention_mask
            combined_attention_mask[:, cc_attention_mask.size(1):] = od_attention_mask[:, 1:]
            
            outputs = self.bert(input_ids=combined_input_ids, attention_mask=combined_attention_mask)
            pooled_output = outputs.pooler_output  # [batch, hidden]
            
            # 简单的线性分类
            logits = torch.matmul(pooled_output, self.knowledge_probes.t())
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
                
                # 简单的线性分类器（需要创建临时的分类头）
                if not hasattr(self, 'temp_classifier'):
                    self.temp_classifier = nn.Linear(self.hidden_size, 141).to(cc_pooled.device)
                
                if self.ablation_type == "w/o_coherence":
                    # 特征拼接（加法）
                    combined_features = cc_pooled + od_pooled
                    logits = self.temp_classifier(combined_features)
                else:
                    # 证据一致性打分（乘法）
                    combined_features = cc_pooled * od_pooled
                    logits = self.temp_classifier(combined_features)
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

def run_inference_ablation_experiment(ablation_type, config, device):
    """运行推理时消融实验"""
    print(f"\n{'='*60}")
    print(f"RUNNING INFERENCE ABLATION EXPERIMENT: {ablation_type}")
    print(f"{'='*60}")
    
    # 加载数据
    label2id, id2label, distance_matrix, class_weights, knowledge_probes = load_preprocessed_data_v13(config)
    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
    
    # 创建数据集
    test_dataset = DualStreamTCMDataset(config.test_path, tokenizer, label2id, config.max_len_cc, config.max_len_od)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    # 创建消融模型
    model = InferenceAblationSyndromeClassifier(config.bert_path, knowledge_probes.to(device), ablation_type)
    model.to(device)
    
    # 加载最优模型权重（只加载兼容的部分）
    best_model_path = os.path.join(config.output_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
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
    
    # 推理时消融实验列表（只测试真正影响推理的组件）
    inference_ablation_types = [
        "full",                    # Ours_Full (完整模型)
        "w/o_coherence",           # 移除证据一致性打分，使用特征拼接
        "w/o_knowledge_probes",    # 移除知识探针，使用BERT pooled output
        "w/o_dual_stream",         # 移除双流，使用单流BERT
    ]
    
    results = {}
    
    for ablation_type in inference_ablation_types:
        try:
            metrics = run_inference_ablation_experiment(ablation_type, args, device)
            results[ablation_type] = metrics
        except Exception as e:
            print(f"Error in {ablation_type}: {e}")
            results[ablation_type] = {"error": str(e)}
    
    # 保存结果
    results_path = os.path.join(args.output_dir, "inference_ablation_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print("INFERENCE ABLATION STUDY RESULTS SUMMARY")
    print(f"{'='*60}")
    
    for ablation_type, metrics in results.items():
        if "error" not in metrics:
            print(f"{ablation_type:20} | Macro-F1: {metrics['macro_f1']:.4f} | AUPRC: {metrics['macro_auprc']:.4f} | AHD: {metrics['ahd']:.4f}")
        else:
            print(f"{ablation_type:20} | ERROR: {metrics['error']}")
    
    print(f"\nResults saved to {results_path}")
    
    print(f"\n{'='*60}")
    print("EXPLANATION OF PREVIOUS RESULTS")
    print(f"{'='*60}")
    print("Previous ablation study had issues:")
    print("1. Loss function ablation (w/o_hls, w/o_focal, w/o_hybrid_loss) was invalid")
    print("   - Loss functions only affect training, not inference")
    print("   - All variants used the same trained model weights")
    print("   - That's why results were identical")
    print("2. Only architecture changes (w/o_coherence, w/o_dual_stream) affect inference")
    print("3. This corrected version focuses on inference-time ablation only")

if __name__ == "__main__":
    main()
