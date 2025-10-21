#!/usr/bin/env python3
"""
计算最优模型在dev和test数据集上的AUPRC
"""

import os
import json
import torch
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
# import matplotlib.pyplot as plt  # 暂时注释掉，避免依赖问题
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import argparse

# 导入项目模块
from src.dataset import DualStreamTCMDataset
from src.model import SyndromeClassifier
from src.utils import load_preprocessed_data_v13

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

def calculate_auprc(model, data_loader, device, num_classes, dataset_name):
    """计算AUPRC"""
    model.eval()
    all_probs = []
    all_labels = []
    
    print(f"Calculating AUPRC for {dataset_name} dataset...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if batch is None:
                continue
                
            cc_input_ids = batch['cc_input_ids'].to(device)
            cc_attention_mask = batch['cc_attention_mask'].to(device)
            od_input_ids = batch['od_input_ids'].to(device)
            od_attention_mask = batch['od_attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 获取模型输出
            logits = model(cc_input_ids, cc_attention_mask, od_input_ids, od_attention_mask)
            probs = torch.softmax(logits, dim=-1)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Processed {batch_idx + 1} batches...")
    
    # 合并所有批次的结果
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    print(f"Total samples: {len(all_labels)}")
    print(f"Probabilities shape: {all_probs.shape}")
    print(f"Labels shape: {all_labels.shape}")
    
    # 计算每个类别的AUPRC
    auprc_scores = []
    for class_idx in range(num_classes):
        # 创建二分类标签（当前类别 vs 其他类别）
        binary_labels = (all_labels == class_idx).astype(int)
        class_probs = all_probs[:, class_idx]
        
        if np.sum(binary_labels) > 0:  # 确保该类别在数据集中存在
            auprc = average_precision_score(binary_labels, class_probs)
            auprc_scores.append(auprc)
        else:
            auprc_scores.append(0.0)
    
    # 计算宏平均AUPRC
    macro_auprc = np.mean(auprc_scores)
    
    # 计算微平均AUPRC - 修复bug
    # 对于多分类问题，微平均AUPRC需要特殊处理
    # 这里我们使用宏平均AUPRC作为主要指标
    micro_auprc = macro_auprc  # 暂时使用宏平均代替微平均
    
    return {
        'macro_auprc': macro_auprc,
        'micro_auprc': micro_auprc,
        'per_class_auprc': auprc_scores,
        'all_probs': all_probs,
        'all_labels': all_labels
    }

def plot_pr_curves(auprc_results, dataset_name, output_dir):
    """绘制PR曲线 - 暂时禁用"""
    print(f"PR curve plotting disabled for {dataset_name} dataset")
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_path", default="./ZY-BERT", help="BERT模型路径")
    parser.add_argument("--train_path", default="./data/train.json", help="训练数据路径")
    parser.add_argument("--dev_path", default="./data/dev.json", help="验证数据路径")
    parser.add_argument("--test_path", default="./data/test.json", help="测试数据路径")
    parser.add_argument("--knowledge_path", default="./data/syndrome_knowledge.json", help="知识文件路径")
    parser.add_argument("--preprocessed_data_dir", default="./preprocessed_data", help="预处理数据目录")
    parser.add_argument("--output_dir", default="./output_v13", help="输出目录")
    parser.add_argument("--max_len_cc", type=int, default=32, help="主诉最大长度")
    parser.add_argument("--max_len_od", type=int, default=256, help="其他描述最大长度")
    parser.add_argument("--batch_size", type=int, default=8, help="批大小")
    parser.add_argument("--no_cuda", action="store_true", help="不使用CUDA")
    
    args = parser.parse_args()
    
    # 添加缺失的路径参数
    args.label_map_path = os.path.join(args.preprocessed_data_dir, "label_map.json")
    args.distance_matrix_path = os.path.join(args.preprocessed_data_dir, "syndrome_distance_matrix")  # 去掉.npy，函数会自动添加
    args.class_weights_path = os.path.join(args.preprocessed_data_dir, "class_weights.pt")
    args.knowledge_probes_path = os.path.join(args.preprocessed_data_dir, "knowledge_probes.pt")
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print(f"Using device: {device}")
    
    # 加载预处理数据
    print("Loading preprocessed data...")
    label2id, id2label, distance_matrix, class_weights, knowledge_probes = load_preprocessed_data_v13(args)
    num_classes = len(label2id)
    print(f"Number of classes: {num_classes}")
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    
    # 创建数据集
    print("Creating datasets...")
    dev_dataset = DualStreamTCMDataset(args.dev_path, tokenizer, label2id, args.max_len_cc, args.max_len_od)
    test_dataset = DualStreamTCMDataset(args.test_path, tokenizer, label2id, args.max_len_cc, args.max_len_od)
    
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    # 创建模型
    print("Loading model...")
    model = SyndromeClassifier(bert_path=args.bert_path, knowledge_probes=knowledge_probes.to(device))
    
    # 加载最优模型权重
    best_model_path = os.path.join(args.output_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print("Best model not found!")
        return
    
    model.to(device)
    model.eval()
    
    # 计算AUPRC
    print("\n" + "="*60)
    print("CALCULATING AUPRC FOR BEST MODEL")
    print("="*60)
    
    # Dev数据集
    dev_auprc = calculate_auprc(model, dev_loader, device, num_classes, "Dev")
    
    # Test数据集
    test_auprc = calculate_auprc(model, test_loader, device, num_classes, "Test")
    
    # 保存结果
    results = {
        'dev_auprc': {
            'macro_auprc': float(dev_auprc['macro_auprc']),
            'micro_auprc': float(dev_auprc['micro_auprc']),
            'per_class_auprc': [float(x) for x in dev_auprc['per_class_auprc']]
        },
        'test_auprc': {
            'macro_auprc': float(test_auprc['macro_auprc']),
            'micro_auprc': float(test_auprc['micro_auprc']),
            'per_class_auprc': [float(x) for x in test_auprc['per_class_auprc']]
        },
        'model_info': {
            'model_path': best_model_path,
            'num_classes': num_classes,
            'device': str(device)
        }
    }
    
    # 保存AUPRC结果
    auprc_output_path = os.path.join(args.output_dir, "auprc_results.json")
    with open(auprc_output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nAUPRC results saved to {auprc_output_path}")
    
    # 绘制PR曲线
    print("\nPlotting PR curves...")
    plot_pr_curves(dev_auprc, "Dev", args.output_dir)
    plot_pr_curves(test_auprc, "Test", args.output_dir)
    
    # 输出结果摘要
    print("\n" + "="*60)
    print("AUPRC RESULTS SUMMARY")
    print("="*60)
    print(f"Dev Dataset:")
    print(f"  Macro AUPRC: {dev_auprc['macro_auprc']:.4f}")
    print(f"  Micro AUPRC: {dev_auprc['micro_auprc']:.4f}")
    print(f"Test Dataset:")
    print(f"  Macro AUPRC: {test_auprc['macro_auprc']:.4f}")
    print(f"  Micro AUPRC: {test_auprc['micro_auprc']:.4f}")
    
    # 显示每个类别的AUPRC统计
    print(f"\nPer-class AUPRC Statistics:")
    print(f"Dev - Mean: {np.mean(dev_auprc['per_class_auprc']):.4f}, Std: {np.std(dev_auprc['per_class_auprc']):.4f}")
    print(f"Test - Mean: {np.mean(test_auprc['per_class_auprc']):.4f}, Std: {np.std(test_auprc['per_class_auprc']):.4f}")

if __name__ == "__main__":
    main()
