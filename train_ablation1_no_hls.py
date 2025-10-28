#!/usr/bin/env python3
"""
Ablation 1: 验证HLS Loss的必要性
Main Model (单流架构) - HLS Loss = 仅Focal Loss
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

from src.single_stream_model import SingleStreamClassifier
from src.single_stream_dataset import SingleStreamTCMDataset
from src.focal_only_loss import FocalOnlyLoss
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
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
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
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    report = classification_report(all_labels, all_preds, 
                                   labels=range(len(id2label)),
                                   output_dict=True, zero_division=0)
    macro_precision = report['macro avg']['precision']
    macro_recall = report['macro avg']['recall']
    
    # AUPRC
    num_labels = len(id2label)
    labels_one_hot = label_binarize(all_labels, classes=list(range(num_labels)))
    all_probs_np = np.array(all_probs)
    auprc = average_precision_score(labels_one_hot, all_probs_np, average='macro')
    
    # AHD
    for true, pred in zip(all_labels, all_preds):
        total_hier_dist += distance_matrix[true][pred]
    avg_hier_dist = total_hier_dist / len(all_labels) if len(all_labels) > 0 else 0
    
    return avg_loss, acc, f1, macro_precision, macro_recall, avg_hier_dist, auprc


def main(args):
    print("="*80)
    print("Ablation 1: Main Model - HLS Loss (Focal Only)")
    print("="*80)
    
    # 加载数据
    print("\nLoading data...")
    data = load_preprocessed_data()
    
    train_dataset = SingleStreamTCMDataset(
        data_path=args.train_data,
        label2id=data['label2id'],
        tokenizer_path=args.bert_path
    )
    
    dev_dataset = SingleStreamTCMDataset(
        data_path=args.dev_data,
        label2id=data['label2id'],
        tokenizer_path=args.bert_path
    )
    
    test_dataset = SingleStreamTCMDataset(
        data_path=args.test_data,
        label2id=data['label2id'],
        tokenizer_path=args.bert_path
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(f"\nDevice: {device}, GPUs: {n_gpu}")
    
    # 初始化模型
    model = SingleStreamClassifier(
        bert_path=args.bert_path,
        knowledge_probes=data['knowledge_probes']
    ).to(device)
    
    if n_gpu > 1:
        model = nn.DataParallel(model)
    
    # 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    # 损失函数（仅Focal Loss）
    criterion = FocalOnlyLoss(
        class_weights=data['class_weights'],
        device=device,
        gamma=args.gamma
    )
    
    # 训练
    best_f1 = 0
    epoch_metrics = []
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        
        # 训练
        model.train()
        for batch in tqdm(train_loader, desc="Training"):
            if batch is None:
                continue
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
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
        
        test_metrics = evaluate(model, test_loader, criterion, device,
                               data['id2label'], data['distance_matrix'].numpy())
        test_loss, test_acc, test_f1, test_p, test_r, test_ahd, test_auprc = test_metrics
        
        print(f"\nDev - Loss: {dev_loss:.4f}, Acc: {dev_acc:.4f}, F1: {dev_f1:.4f}, "
              f"P: {dev_p:.4f}, R: {dev_r:.4f}, AHD: {dev_ahd:.4f}, AUPRC: {dev_auprc:.4f}")
        print(f"Test - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}, "
              f"P: {test_p:.4f}, R: {test_r:.4f}, AHD: {test_ahd:.4f}, AUPRC: {test_auprc:.4f}")
        
        # 记录指标
        epoch_metrics.append({
            'epoch': epoch + 1,
            'dev_loss': dev_loss,
            'dev_accuracy': dev_acc,
            'dev_macro_f1': dev_f1,
            'dev_macro_precision': dev_p,
            'dev_macro_recall': dev_r,
            'dev_avg_hier_distance': dev_ahd,
            'dev_auprc': dev_auprc,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_macro_f1': test_f1,
            'test_macro_precision': test_p,
            'test_macro_recall': test_r,
            'test_avg_hier_distance': test_ahd,
            'test_auprc': test_auprc
        })
        
        # 保存最佳模型
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                      os.path.join(args.output_dir, 'best_model.pt'))
            print(f"\nNew best model saved! Dev F1: {best_f1:.4f}")
    
    # 保存所有epoch的指标
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'all_epochs_metrics.json'), 'w') as f:
        json.dump(epoch_metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print("Training completed!")
    print(f"Best Dev F1: {best_f1:.4f}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="data/train.json")
    parser.add_argument("--dev_data", type=str, default="data/dev.json")
    parser.add_argument("--test_data", type=str, default="data/test.json")
    parser.add_argument("--bert_path", type=str, default="./ZY-BERT")
    parser.add_argument("--output_dir", type=str, default="output_ablation1_no_hls")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--gamma", type=float, default=2.0)
    
    args = parser.parse_args()
    main(args)

