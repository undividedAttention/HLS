import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.utils import preprocess_and_save, load_preprocessed_data
from src.dataset import TCMDataset
from src.model import SyndromeClassifier
from src.loss import HLSLoss

def train(model, train_loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(logits, labels)
        
        # 如果使用了DataParallel，loss会是一个向量，需要取平均值
        if isinstance(loss, torch.Tensor) and loss.dim() > 0:
            loss = loss.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(train_loader)

def evaluate(model, data_loader, criterion, device, id2label, distance_matrix):
    model.eval()
    total_loss = 0
    total_hier_dist = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)

            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            
            for i in range(len(labels)):
                true_label_id = labels[i].item()
                pred_label_id = preds[i].item()
                total_hier_dist += distance_matrix[true_label_id, pred_label_id]

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    avg_hier_dist = total_hier_dist / len(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    unique_labels_in_data = np.unique(all_labels + all_preds)
    # id2label的键可能是字符串，统一转换
    target_names = [id2label[str(i)] for i in unique_labels_in_data if str(i) in id2label]
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4, zero_division=0)

    return avg_loss, accuracy, f1, avg_hier_dist, report


def main(config):
    # --- 1. 设置设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count() if str(device) == "cuda" else 0
    
    if str(device) == "cuda":
        print(f"检测到 {n_gpu} 块GPU，将使用CUDA进行加速。")
    else:
        print("未检测到GPU或已指定不使用GPU，将使用CPU进行训练。")

    if config.do_preprocess:
        preprocess_and_save(config)
    
    label2id, id2label, distance_matrix = load_preprocessed_data(config)
    distance_matrix_np = distance_matrix.numpy()
    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
    num_labels = len(label2id)

    # --- 2. 加载数据集 ---
    print("Loading datasets...")
    train_dataset = TCMDataset(config.train_path, tokenizer, label2id, config.max_seq_len)
    dev_dataset = TCMDataset(config.dev_path, tokenizer, label2id, config.max_seq_len)
    test_dataset = TCMDataset(config.test_path, tokenizer, label2id, config.max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    
    # --- 3. 初始化模型并应用DataParallel ---
    print("Initializing model...")
    model = SyndromeClassifier(config.bert_path, num_labels)
    model.to(device)

    if n_gpu > 1:
        print(f"使用 {n_gpu} 块GPU进行Data Parallel训练...")
        model = nn.DataParallel(model)

    criterion = HLSLoss(distance_matrix, temperature=config.temperature, device=device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, eps=1e-8)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)

    # --- 4. 训练与评估循环 ---
    best_f1 = 0
    for epoch in range(config.epochs):
        print(f"\n--- Epoch {epoch+1}/{config.epochs} ---")
        train_loss = train(model, train_loader, optimizer, scheduler, criterion, device)
        print(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}")
        
        dev_loss, dev_acc, dev_f1, dev_ahd, dev_report = evaluate(model, dev_loader, criterion, device, id2label, distance_matrix_np)
        print(f"Epoch {epoch+1} Dev Loss: {dev_loss:.4f}, Accuracy: {dev_acc:.4f}, Macro-F1: {dev_f1:.4f}, Avg Hier Distance: {dev_ahd:.4f}")

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            os.makedirs(config.output_dir, exist_ok=True)
            
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), os.path.join(config.output_dir, "best_model.pt"))
            print(f"New best model saved with F1: {best_f1:.4f}")

    # --- 5. 最终测试 ---
    print("\n--- Final Evaluation on Test Set using Best Model ---")
    final_model = SyndromeClassifier(config.bert_path, num_labels)
    final_model.load_state_dict(torch.load(os.path.join(config.output_dir, "best_model.pt")))
    final_model.to(device)
    
    if n_gpu > 1:
        final_model = nn.DataParallel(final_model)

    test_loss, test_acc, test_f1, test_ahd, test_report = evaluate(final_model, test_loader, criterion, device, id2label, distance_matrix_np)
    print(f"\nTest Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, Macro-F1: {test_f1:.4f}, Avg Hier Distance: {test_ahd:.4f}")
    print("\n--- Test Set Classification Report ---")
    print(test_report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train_path", type=str, default="./data/train.json")
    parser.add_argument("--dev_path", type=str, default="./data/dev.json")
    parser.add_argument("--test_path", type=str, default="./data/test.json")
    parser.add_argument("--bert_path", type=str, default="./ZY-BERT")
    parser.add_argument("--syndrome_path_file", type=str, default="./data/syndromeToPath.txt")
    parser.add_argument("--vocab_path", type=str, default="./data/vocabulary.txt")
    parser.add_argument("--preprocessed_data_dir", type=str, default="./preprocessed_data")
    parser.add_argument("--output_dir", type=str, default="./output")
    
    parser.add_argument("--do_preprocess", action='store_true', help="如果指定，则执行数据预处理步骤")
    parser.add_argument("--no_cuda", action='store_true', help="如果指定，则不使用GPU，即使有可用的GPU")

    parser.add_argument("--max_seq_len", type=int, default=128)
    # 增加batch_size的默认值，并添加注释
    parser.add_argument("--batch_size", type=int, default=16, help="总的batch size，会被自动分配到所有可用GPU上")
    # 增加epochs的默认值
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    # 将temperature的默认值改回1.0
    parser.add_argument("--temperature", type=float, default=1.0, help="HLS Loss的温度参数")
    
    args = parser.parse_args()

    args.label_map_path = os.path.join(args.preprocessed_data_dir, "label_map.json")
    args.graph_path = os.path.join(args.preprocessed_data_dir, "syndrome_knowledge_graph.gml")
    args.distance_matrix_path = os.path.join(args.preprocessed_data_dir, "syndrome_distance_matrix")

    main(args)

