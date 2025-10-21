import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.utils import preprocess_and_save, load_preprocessed_data_v13, regenerate_assets
from src.dataset import DualStreamTCMDataset
from src.model import SyndromeClassifier
from src.loss import HybridLoss

def collate_fn(batch):
    """过滤掉Dataset返回的None值。"""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def train(model, train_loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for batch in progress_bar:
        if batch is None: continue
        optimizer.zero_grad()
        logits = model(
            cc_input_ids=batch['cc_input_ids'].to(device),
            cc_attention_mask=batch['cc_attention_mask'].to(device),
            od_input_ids=batch['od_input_ids'].to(device),
            od_attention_mask=batch['od_attention_mask'].to(device)
        )
        labels = batch['labels'].to(device)
        loss = criterion(logits, labels)
        
        if hasattr(model, 'module'): loss = loss.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(train_loader) if len(train_loader) > 0 else 0

def evaluate(model, data_loader, criterion, device, id2label, distance_matrix):
    model.eval()
    total_loss = 0
    total_hier_dist = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            if batch is None: continue
            logits = model(
                cc_input_ids=batch['cc_input_ids'].to(device),
                cc_attention_mask=batch['cc_attention_mask'].to(device),
                od_input_ids=batch['od_input_ids'].to(device),
                od_attention_mask=batch['od_attention_mask'].to(device)
            )
            labels = batch['labels'].to(device)
            loss = criterion(logits, labels)

            if hasattr(model, 'module'): loss = loss.mean()
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            
            for i in range(len(labels)):
                total_hier_dist += distance_matrix[labels[i].item(), preds[i].item()]

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if not all_labels:
        return 0, 0, 0, 0, "没有样本可供评估。"
        
    avg_loss = total_loss / len(data_loader)
    avg_hier_dist = total_hier_dist / len(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    unique_labels = np.unique(all_labels + all_preds)
    target_names = [id2label.get(str(i)) for i in unique_labels if str(i) in id2label]
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4, zero_division=0)

    return avg_loss, accuracy, f1, avg_hier_dist, report

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count() if str(device) == "cuda" else 0
    
    if config.do_preprocess:
        preprocess_and_save(config)
    
    label2id, id2label, distance_matrix, class_weights, knowledge_probes = load_preprocessed_data_v13(config)
    
    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
    
    train_dataset = DualStreamTCMDataset(config.train_path, tokenizer, label2id, config.max_len_cc, config.max_len_od)
    dev_dataset = DualStreamTCMDataset(config.dev_path, tokenizer, label2id, config.max_len_cc, config.max_len_od)
    test_dataset = DualStreamTCMDataset(config.test_path, tokenizer, label2id, config.max_len_cc, config.max_len_od)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    model = SyndromeClassifier(bert_path=config.bert_path, knowledge_probes=knowledge_probes.to(device))
    model.to(device)
    if n_gpu > 1: model = nn.DataParallel(model)
    
    criterion = HybridLoss(distance_matrix, class_weights, device, config.lambda_focal, config.lambda_hls, config.gamma, config.temperature_hls)
    
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, eps=1e-8)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)

    best_f1 = 0
    epoch_metrics = []
    for epoch in range(config.epochs):
        print(f"\n--- Epoch {epoch+1}/{config.epochs} ---")
        train(model, train_loader, optimizer, scheduler, criterion, device)
        dev_loss, dev_acc, dev_f1, dev_ahd, _ = evaluate(model, dev_loader, criterion, device, id2label, distance_matrix.numpy())
        test_loss, test_acc, test_f1, test_ahd, _ = evaluate(model, test_loader, criterion, device, id2label, distance_matrix.numpy())
        print(f"Epoch {epoch+1} Dev Loss: {dev_loss:.4f}, Acc: {dev_acc:.4f}, Macro-F1: {dev_f1:.4f}, AHD: {dev_ahd:.4f}")
        print(f"Epoch {epoch+1} Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Macro-F1: {test_f1:.4f}, AHD: {test_ahd:.4f}")

        epoch_metrics.append({
            'epoch': epoch+1,
            'dev_loss': dev_loss,
            'dev_accuracy': dev_acc,
            'dev_macro_f1': dev_f1,
            'dev_avg_hier_distance': dev_ahd,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_macro_f1': test_f1,
            'test_avg_hier_distance': test_ahd,
        })

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            os.makedirs(config.output_dir, exist_ok=True)
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), os.path.join(config.output_dir, "best_model.pt"))
            print(f"New best model saved with F1: {best_f1:.4f}")

        # 持续写入所有epoch指标
        try:
            os.makedirs(config.output_dir, exist_ok=True)
            import json as _json
            with open(os.path.join(config.output_dir, 'all_epochs_metrics.json'), 'w', encoding='utf-8') as f:
                _json.dump(epoch_metrics, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to write epoch metrics: {e}")

    print("\n--- Final Evaluation on Test Set ---")
    final_model = SyndromeClassifier(config.bert_path, knowledge_probes.to(device))
    map_location = device if str(device) == "cpu" else None
    final_model.load_state_dict(torch.load(os.path.join(config.output_dir, "best_model.pt"), map_location=map_location))
    final_model.to(device)
    if n_gpu > 1: final_model = nn.DataParallel(final_model)
    
    _, test_acc, test_f1, test_ahd, test_report = evaluate(final_model, test_loader, criterion, device, id2label, distance_matrix.numpy())
    print(f"\nTest Acc: {test_acc:.4f}, Macro-F1: {test_f1:.4f}, AHD: {test_ahd:.4f}")
    print("\n--- Test Set Classification Report ---\n", test_report)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # 文件路径
    parser.add_argument("--train_path", default="./data/train.json")
    parser.add_argument("--dev_path", default="./data/dev.json")
    parser.add_argument("--test_path", default="./data/test.json")
    parser.add_argument("--bert_path", default="./ZY-BERT")
    parser.add_argument("--vocab_path", default="./data/vocabulary.txt")
    parser.add_argument("--syndrome_path_file", default="./data/syndromeToPath.txt")
    parser.add_argument("--knowledge_path", default="./data/syndrome_knowledge.json", help="V13新增：知识定义文件路径")
    parser.add_argument("--preprocessed_data_dir", default="./preprocessed_data")
    parser.add_argument("--output_dir", default="./output_v13")
    
    # 控制开关
    parser.add_argument("--do_preprocess", action='store_true', help="执行所有预处理步骤，包括生成知识探针")
    parser.add_argument("--regen_assets", action='store_true', help="Regenerate class_weights and knowledge_probes from data")
    parser.add_argument("--no_cuda", action='store_true')

    # 模型超参数
    parser.add_argument("--max_len_cc", type=int, default=32, help="主诉最大长度")
    parser.add_argument("--max_len_od", type=int, default=256, help="其他描述最大长度")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    
    # 损失函数超参数
    parser.add_argument("--lambda_focal", type=float, default=0.7, help="Focal Loss的权重")
    parser.add_argument("--lambda_hls", type=float, default=0.3, help="HLS Loss的权重")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal Loss的gamma参数")
    parser.add_argument("--temperature_hls", type=float, default=1.0, help="HLS Loss的温度参数")

    args = parser.parse_args()

    # 自动生成预处理文件的路径
    args.label_map_path = os.path.join(args.preprocessed_data_dir, "label_map.json")
    args.distance_matrix_path = os.path.join(args.preprocessed_data_dir, "syndrome_distance_matrix")
    args.class_weights_path = os.path.join(args.preprocessed_data_dir, "class_weights.pt")
    args.knowledge_probes_path = os.path.join(args.preprocessed_data_dir, "knowledge_probes.pt")
    args.syndrome_knowledge_path = os.path.join("./data", "syndrome_knowledge.json")

    # 若需要，先构建/刷新 assets（依赖label_map）
    if args.regen_assets:
        with open(args.label_map_path, 'r', encoding='utf-8') as f:
            label_map_tmp = json.load(f)
        regenerate_assets(args, label_map_tmp['label2id'])

    main(args)

