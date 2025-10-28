#!/usr/bin/env python3
"""
网格搜索寻找最优超参数配置
使用单流架构 + HybridLoss作为Main Model
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, average_precision_score
from sklearn.preprocessing import label_binarize
from src.model import SyndromeClassifier
from src.single_stream_model import SingleStreamClassifier
from src.single_stream_dataset import SingleStreamTCMDataset
from src.loss import HybridLoss
from src.utils import load_preprocessed_data


def evaluate(model, data_loader, criterion, device, id2label, distance_matrix):
    model.eval()
    total_loss = 0
    total_hier_dist = 0
    all_preds, all_labels = [], []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            if batch is None:
                continue
            
            input_ids = batch['concatenated_input_ids'].to(device)
            attention_mask = batch['concatenated_attention_mask'].to(device)
            
            logits = model(concatenated_input_ids=input_ids, concatenated_attention_mask=attention_mask)
            labels = batch['labels'].to(device)
            loss = criterion(logits, labels)

            if hasattr(model, 'module'):
                loss = loss.mean()
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            
            for i in range(len(labels)):
                true_label = int(labels[i].item())
                pred_label = int(preds[i].item())
                all_preds.append(pred_label)
                all_labels.append(true_label)
                all_probs.append(probs[i].cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    
    # 计算准确率
    acc = accuracy_score(all_labels, all_preds)
    
    # 计算Macro-F1
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    # 计算Macro-Precision和Macro-Recall
    report = classification_report(all_labels, all_preds, 
                                   labels=range(len(id2label)),
                                   output_dict=True, zero_division=0)
    macro_precision = report['macro avg']['precision']
    macro_recall = report['macro avg']['recall']
    
    # 计算AUPRC
    num_labels = len(id2label)
    labels_one_hot = label_binarize(all_labels, classes=list(range(num_labels)))
    all_probs_np = np.array(all_probs)
    auprc = average_precision_score(labels_one_hot, all_probs_np, average='macro')
    
    # 计算平均层次距离（AHD）
    for true, pred in zip(all_labels, all_preds):
        total_hier_dist += distance_matrix[true][pred]
    avg_hier_dist = total_hier_dist / len(all_labels) if len(all_labels) > 0 else 0
    
    return avg_loss, acc, f1, macro_precision, macro_recall, avg_hier_dist, auprc


def run_grid_search(args, search_space):
    """运行网格搜索"""
    
    # 加载数据
    print("Loading data...")
    data = load_preprocessed_data()
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    
    results = []
    
    # 遍历所有参数组合
    for lambda_focal in search_space['lambda_focal']:
        for lambda_hls in search_space['lambda_hls']:
            for lr in search_space['learning_rate']:
                for gamma in search_space['gamma']:
                    
                    config_name = f"lf{lambda_focal}_lh{lambda_hls}_lr{lr}_g{gamma}"
                    print(f"\n{'='*80}")
                    print(f"Config: {config_name}")
                    print(f"{'='*80}")
                    
                    # 创建数据集（每个配置重新创建，避免内存问题）
                    train_dataset = SingleStreamTCMDataset(
                        file_path=args.train_data,
                        tokenizer=tokenizer,
                        label2id=data['label2id']
                    )
                    
                    dev_dataset = SingleStreamTCMDataset(
                        file_path=args.dev_data,
                        tokenizer=tokenizer,
                        label2id=data['label2id']
                    )
                    
                    # 创建数据加载器
                    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
                    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
                    
                    # 初始化模型
                    model = SingleStreamClassifier(
                        bert_path=args.bert_path,
                        knowledge_probes=data['knowledge_probes']
                    ).to(device)
                    
                    if n_gpu > 1:
                        model = nn.DataParallel(model)
                    
                    # 设置优化器和学习率调度器
                    optimizer = AdamW(model.parameters(), lr=lr)
                    
                    total_steps = len(train_loader) * args.epochs
                    scheduler = get_linear_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=0,
                        num_training_steps=total_steps
                    )
                    
                    # 创建损失函数
                    criterion = HybridLoss(
                        distance_matrix=data['distance_matrix'],
                        class_weights=data['class_weights'],
                        device=device,
                        lambda_focal=lambda_focal,
                        lambda_hls=lambda_hls,
                        gamma=gamma
                    )
                    
                    # 训练并记录所有epoch的指标
                    all_epoch_metrics = []
                    best_dev_f1 = 0
                    best_epoch = 0
                    
                    for epoch in range(args.epochs):
                        print(f"\nEpoch {epoch+1}/{args.epochs}")
                        
                        # 训练
                        model.train()
                        for batch in tqdm(train_loader, desc="Training", leave=False):
                            if batch is None:
                                continue
                            
                            input_ids = batch['concatenated_input_ids'].to(device)
                            attention_mask = batch['concatenated_attention_mask'].to(device)
                            labels = batch['labels'].to(device)
                            
                            logits = model(concatenated_input_ids=input_ids, concatenated_attention_mask=attention_mask)
                            loss = criterion(logits, labels)
                            
                            if hasattr(model, 'module'):
                                loss = loss.mean()
                            
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                        
                        # 评估
                        dev_metrics = evaluate(model, dev_loader, criterion, device, 
                                              data['id2label'], data['distance_matrix'].numpy())
                        dev_loss, dev_acc, dev_f1, dev_p, dev_r, dev_ahd, dev_auprc = dev_metrics
                        
                        print(f"Dev - Loss: {dev_loss:.4f}, Acc: {dev_acc:.4f}, "
                              f"F1: {dev_f1:.4f}, P: {dev_p:.4f}, R: {dev_r:.4f}, "
                              f"AHD: {dev_ahd:.4f}, AUPRC: {dev_auprc:.4f}")
                        
                        # 记录每个epoch的指标
                        epoch_metrics = {
                            'epoch': epoch + 1,
                            'dev_loss': dev_loss,
                            'dev_accuracy': dev_acc,
                            'dev_macro_f1': dev_f1,
                            'dev_macro_precision': dev_p,
                            'dev_macro_recall': dev_r,
                            'dev_avg_hier_distance': dev_ahd,
                            'dev_auprc': dev_auprc
                        }
                        all_epoch_metrics.append(epoch_metrics)
                        
                        # 跟踪最佳结果
                        if dev_f1 > best_dev_f1:
                            best_dev_f1 = dev_f1
                            best_epoch = epoch + 1
                    
                    # 保存最佳配置的结果
                    best_metrics = all_epoch_metrics[best_epoch - 1].copy()
                    best_metrics['config'] = config_name
                    best_metrics['lambda_focal'] = lambda_focal
                    best_metrics['lambda_hls'] = lambda_hls
                    best_metrics['learning_rate'] = lr
                    best_metrics['gamma'] = gamma
                    
                    results.append(best_metrics)
                    print(f"\nBest Dev F1: {best_dev_f1:.4f} at Epoch {best_epoch}")
    
    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    results_file = os.path.join(args.output_dir, 'grid_search_results.json')
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 找到最佳配置
    best_config = max(results, key=lambda x: x['dev_macro_f1'])
    print(f"\n{'='*80}")
    print("Best Configuration:")
    print(f"Lambda Focal: {best_config['lambda_focal']}")
    print(f"Lambda HLS: {best_config['lambda_hls']}")
    print(f"Learning Rate: {best_config['learning_rate']}")
    print(f"Gamma: {best_config['gamma']}")
    print(f"Best Dev F1: {best_config['dev_macro_f1']:.4f}")
    print(f"{'='*80}\n")
    
    # 保存最佳配置
    best_config_file = os.path.join(args.output_dir, 'best_config.json')
    with open(best_config_file, 'w') as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="data/train.json")
    parser.add_argument("--dev_data", type=str, default="data/dev.json")
    parser.add_argument("--test_data", type=str, default="data/test.json")
    parser.add_argument("--bert_path", type=str, default="./ZY-BERT")
    parser.add_argument("--output_dir", type=str, default="output_grid_search")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)  # 训练足够多的epochs以找到最佳结果
    
    args = parser.parse_args()
    
    # 定义搜索空间
    search_space = {
        'lambda_focal': [0.5, 0.7, 0.9],
        'lambda_hls': [0.1, 0.3, 0.5],
        'learning_rate': [1e-5, 2e-5, 3e-5],
        'gamma': [1.0, 2.0, 3.0]
    }
    
    print("Starting Grid Search...")
    print(f"Search space: {search_space}")
    print(f"Total combinations: {len(search_space['lambda_focal']) * len(search_space['lambda_hls']) * len(search_space['learning_rate']) * len(search_space['gamma'])}")
    
    run_grid_search(args, search_space)

