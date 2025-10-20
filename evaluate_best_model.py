#!/usr/bin/env python3
"""
使用双流实验的最优模型进行dev和test数据评估
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.utils import load_preprocessed_data_v11
from src.dataset import DualStreamTCMDataset
from src.model import SyndromeClassifier
from src.loss import HybridLoss

def collate_fn(batch):
    """自定义的collate_fn，用于过滤掉Dataset返回的None值。"""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def evaluate_model(model, data_loader, criterion, distance_matrix, device, dataset_name):
    """评估模型性能"""
    model.eval()
    total_loss = 0
    total_hier_dist = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {dataset_name}", leave=False):
            if batch is None: continue
            logits = model(
                cc_input_ids=batch['cc_input_ids'].to(device),
                cc_attention_mask=batch['cc_attention_mask'].to(device),
                od_input_ids=batch['od_input_ids'].to(device),
                od_attention_mask=batch['od_attention_mask'].to(device)
            )
            labels = batch['label'].to(device)
            loss = criterion(logits, labels)

            if hasattr(model, 'module'): loss = loss.mean()
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            
            for i in range(len(labels)):
                total_hier_dist += distance_matrix[labels[i].item(), preds[i].item()]

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    avg_hier_dist = total_hier_dist / len(all_labels)

    print(f"\n--- {dataset_name} Set Results ---")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Avg Hier Distance: {avg_hier_dist:.4f}")
    
    # 生成详细分类报告
    print(f"\n--- {dataset_name} Set Classification Report ---")
    report = classification_report(all_labels, all_preds, target_names=None, output_dict=False)
    print(report)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'avg_hier_distance': avg_hier_dist,
        'predictions': all_preds,
        'labels': all_labels
    }

def main():
    # 配置参数
    config = type('Config', (), {
        'preprocessed_data_dir': './preprocessed_data',
        'bert_path': './ZY-BERT',
        'train_path': './data/train.json',
        'dev_path': './data/dev.json', 
        'test_path': './data/test.json',
        'max_len_cc': 32,
        'max_len_od': 256,
        'batch_size': 16,
        'lambda_focal': 0.7,
        'lambda_hls': 0.3,
        'gamma': 2.0,
        'temperature': 1.0
    })()
    
    # 设置路径
    config.label_map_path = os.path.join(config.preprocessed_data_dir, "label_map_v11.2.json")
    config.distance_matrix_path = os.path.join(config.preprocessed_data_dir, "distance_matrix_v11.2")
    config.class_weights_path = os.path.join(config.preprocessed_data_dir, "class_weights_v11.2.pt")
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载预处理数据
    print("Loading preprocessed data...")
    label2id, id2label, distance_matrix, class_weights = load_preprocessed_data_v11(config)
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
    
    # 创建数据集
    print("Creating datasets...")
    dev_dataset = DualStreamTCMDataset(config.dev_path, tokenizer, label2id, config.max_len_cc, config.max_len_od)
    test_dataset = DualStreamTCMDataset(config.test_path, tokenizer, label2id, config.max_len_cc, config.max_len_od)
    
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    # 创建模型
    print("Loading model...")
    model = SyndromeClassifier(bert_path=config.bert_path, num_labels=len(label2id))
    
    # 加载最优模型权重
    best_model_path = './output_dual_stream/best_model.pt'
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint)
    else:
        print("Best model not found, using random weights")
    
    model.to(device)
    
    # 创建损失函数
    criterion = HybridLoss(distance_matrix, torch.ones(len(label2id)), device, 
                          lambda_focal=config.lambda_focal, lambda_hls=config.lambda_hls,
                          gamma=config.gamma, temperature_hls=config.temperature)
    
    # 评估模型
    print("\n" + "="*50)
    print("EVALUATING BEST MODEL FROM DUAL STREAM EXPERIMENT")
    print("="*50)
    
    # 评估dev集
    dev_results = evaluate_model(model, dev_loader, criterion, distance_matrix, device, "Dev")
    
    # 评估test集
    test_results = evaluate_model(model, test_loader, criterion, distance_matrix, device, "Test")
    
    # 保存结果
    results = {
        'dev_results': dev_results,
        'test_results': test_results,
        'model_info': {
            'model_path': best_model_path,
            'num_labels': len(label2id),
            'device': str(device)
        }
    }
    
    with open('./output_dual_stream/best_model_evaluation.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to ./output_dual_stream/best_model_evaluation.json")
    
    # 总结
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Dev Set - Accuracy: {dev_results['accuracy']:.4f}, Macro-F1: {dev_results['macro_f1']:.4f}, Avg Hier Distance: {dev_results['avg_hier_distance']:.4f}")
    print(f"Test Set - Accuracy: {test_results['accuracy']:.4f}, Macro-F1: {test_results['macro_f1']:.4f}, Avg Hier Distance: {test_results['avg_hier_distance']:.4f}")

if __name__ == "__main__":
    main()
