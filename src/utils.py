import os
import networkx as nx
import json
import re
import numpy as np
import torch
from tqdm import tqdm
from itertools import combinations
from collections import Counter
from transformers import BertModel, AutoTokenizer

def get_synonym_map():
    """定义同义词到标准词的映射。"""
    return {
        "风寒湿阻证": "风寒湿痹证", "风痰闭阻证": "风痰闭窍证", "热毒壅结证": "热毒壅结证",
        "气虚血溢证": "气不摄血证", "气血不足证": "气血亏虚证", "肝肾亏损证": "肝肾亏虚证",
        "脾胃虚寒证": "脾胃阳虚证"
    }

def create_knowledge_probes(config, label2id, device):
    """
    V13核心创新：从syndrome_knowledge.json生成并保存知识探针。
    """
    print("--- 正在创建知识探针 (Knowledge Probes) ---")
    
    # 加载知识文件
    try:
        knowledge_data = []
        with open(config.knowledge_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                knowledge_data.append(json.loads(line))
    except FileNotFoundError:
        print(f"错误: 知识文件未找到于 {config.knowledge_path}")
        return

    # 创建一个从标准标签名到知识文本的映射
    knowledge_map = {item['Name']: item.get('Definition', '') + " " + item.get('Typical_performance', '') 
                     for item in knowledge_data}

    # 加载预训练模型用于编码
    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
    bert_model = BertModel.from_pretrained(config.bert_path).to(device)
    bert_model.eval()

    num_labels = len(label2id)
    hidden_size = bert_model.config.hidden_size
    knowledge_probes = torch.zeros(num_labels, hidden_size)

    with torch.no_grad():
        for label, idx in tqdm(label2id.items(), desc="编码知识探针"):
            knowledge_text = knowledge_map.get(label)
            if not knowledge_text:
                print(f"警告: 标签 '{label}' 在知识文件中没有找到定义。将使用零向量。")
                continue
            
            inputs = tokenizer.encode_plus(
                knowledge_text,
                add_special_tokens=True,
                max_length=512, # 使用足够长的序列来编码定义
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(device)
            
            outputs = bert_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            cls_embedding = outputs.pooler_output
            knowledge_probes[idx] = cls_embedding.squeeze(0).cpu()

    torch.save(knowledge_probes, config.knowledge_probes_path)
    print(f"知识探针已创建并保存至: {config.knowledge_probes_path}")


def parse_syndrome_paths(filepath: str, synonym_map: dict) -> dict:
    """解析路径文件，并使用标准名称。"""
    # (此函数与V11.2版本一致, 为保持完整性而包含)
    paths = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_syndrome = None
    for line in lines:
        line = line.strip()
        if not line: continue
        
        syndrome_match = re.match(r'症候:\s*(.*)', line)
        if syndrome_match:
            current_syndrome = syndrome_match.group(1).strip()
            current_syndrome = synonym_map.get(current_syndrome, current_syndrome)
            continue
            
        path_match = re.match(r'路径:\s*(.*)', line)
        if path_match and current_syndrome:
            path_str = path_match.group(1).strip()
            path_list = [node.strip() for node in path_str.split('->')]
            standard_path = [synonym_map.get(node, node) for node in path_list]
            paths[current_syndrome] = standard_path
            current_syndrome = None
            
    isolated_nodes = ["风痰瘀阻证", "正虚毒瘀证"]
    for node in isolated_nodes:
        node = synonym_map.get(node, node)
        if node not in paths:
            paths[node] = [node]
            
    return paths

def build_knowledge_graph(paths: dict, all_labels: list):
    """构建知识图谱，用于计算距离。"""
    # (此函数与V11.2版本一致, 为保持完整性而包含)
    G = nx.DiGraph()
    ROOT_NODE = "ROOT"
    G.add_node(ROOT_NODE)

    all_nodes_in_paths = {ROOT_NODE}
    for path in paths.values():
        all_nodes_in_paths.update(path)
    
    all_graph_nodes = all_nodes_in_paths.union(set(all_labels))
    for node in all_graph_nodes:
        G.add_node(node)

    for syndrome, path in paths.items():
        if not path: continue
        G.add_edge(ROOT_NODE, path[0])
        for i in range(len(path) - 1):
            G.add_edge(path[i], path[i+1])

    for label in all_labels:
        if G.in_degree(label) == 0 and label != ROOT_NODE:
            G.add_edge(ROOT_NODE, label)
            
    return G

def calculate_class_weights(train_path, label2id):
    """计算用于Focal Loss的类别权重。"""
    # (此函数与V11.2版本一致, 为保持完整性而包含)
    print("正在计算类别权重...")
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    label_counts = Counter(item['merge_syndrome'] for item in train_data if item.get('merge_syndrome') in label2id)
    
    weights = torch.zeros(len(label2id))
    for label, count in label_counts.items():
        weights[label2id[label]] = count
    
    beta = 0.999
    effective_num = 1.0 - np.power(beta, weights.numpy())
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(label2id)
    
    print("类别权重计算完成。")
    return torch.FloatTensor(per_cls_weights)

def preprocess_and_save(config):
    """V13 的完整预处理流程。"""
    print("--- 开始V13 (K-FEN) 预处理 ---")
    os.makedirs(config.preprocessed_data_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu")
    
    with open(config.vocab_path, 'r', encoding='utf-8') as f:
        all_labels = sorted([line.strip() for line in f if line.strip()])
    
    label2id = {label: i for i, label in enumerate(all_labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    # 1. 创建并保存知识探针
    create_knowledge_probes(config, label2id, device)

    # 2. 创建并保存类别权重
    class_weights = calculate_class_weights(config.train_path, label2id)
    torch.save(class_weights, config.class_weights_path)
    
    # 3. 创建并保存层次距离矩阵
    synonym_map = get_synonym_map()
    syndrome_paths = parse_syndrome_paths(config.syndrome_path_file, synonym_map)
    graph = build_knowledge_graph(syndrome_paths, all_labels)
    
    depths = nx.shortest_path_length(graph, source="ROOT")
    distance_matrix = np.zeros((len(all_labels), len(all_labels)), dtype=np.float32)
    for i, j in tqdm(list(combinations(range(len(all_labels)), 2)), desc="计算距离矩阵"):
        label1, label2 = id2label[i], id2label[j]
        try:
            lca = nx.lowest_common_ancestor(graph, label1, label2)
            dist = depths.get(label1, 0) + depths.get(label2, 0) - 2 * depths.get(lca, 0)
            distance_matrix[i, j] = distance_matrix[j, i] = dist
        except (nx.NetworkXError, KeyError):
            continue
    np.save(config.distance_matrix_path, distance_matrix)
    
    # 4. 保存标签映射
    label_map = {'label2id': label2id, 'id2label': id2label}
    with open(config.label_map_path, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=4)
        
    print(f"--- V13 预处理完成 ---")

def load_preprocessed_data_v13(config):
    """加载V13预处理好的所有数据。"""
    with open(config.label_map_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    distance_matrix = np.load(config.distance_matrix_path + '.npy')
    class_weights = torch.load(config.class_weights_path)
    knowledge_probes = torch.load(config.knowledge_probes_path)
    return label_map['label2id'], label_map['id2label'], torch.from_numpy(distance_matrix), class_weights, knowledge_probes


# =============== New utilities for regenerating class_weights and knowledge_probes ===============
def compute_class_weights_from_train(train_path: str, label2id: dict) -> torch.Tensor:
    """根据训练集类别频次计算类别权重（inverse frequency）。
    数据格式为JSONL，每行一个样本，其中标签字段为 merge_syndrome。
    """
    class_counts = [0] * len(label2id)
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            label_str = sample.get('merge_syndrome')
            if label_str in label2id:
                class_counts[label2id[label_str]] += 1

    counts = np.array(class_counts, dtype=np.float32)
    # 防止除零
    counts[counts == 0] = 1.0
    inv_freq = 1.0 / counts
    # 归一化到均值为1，稳定训练尺度
    inv_freq = inv_freq * (len(inv_freq) / inv_freq.sum())
    return torch.tensor(inv_freq, dtype=torch.float32)


def build_knowledge_probes_from_knowledge(knowledge_json_path: str, label2id: dict, tokenizer, hidden_size: int) -> torch.Tensor:
    """基于syndrome_knowledge.json中每个标签的 Definition + Typical_performance 文本，
    用同一BERT编码得到每个标签的语义原型（[CLS]向量）。
    knowledge文件结构应能通过标签中文名匹配到条目。
    """
    # 兼容JSON或JSONL两种格式
    with open(knowledge_json_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    knowledge = {}
    try:
        obj = json.loads(content)
        if isinstance(obj, dict):
            knowledge = obj
        elif isinstance(obj, list):
            # 如果是列表，尝试每项包含label_name键
            for entry in obj:
                if not isinstance(entry, dict):
                    continue
                name = entry.get('label_name') or entry.get('name') or entry.get('label')
                if not name:
                    continue
                knowledge[name] = entry
        else:
            knowledge = {}
    except json.JSONDecodeError:
        # 按JSONL逐行读取
        knowledge = {}
        with open(knowledge_json_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                name = entry.get('label_name') or entry.get('name') or entry.get('label')
                if name:
                    knowledge[name] = entry

    # 构建从标签名到文本的映射
    # 期望结构：{ label_name: { "Definition": str, "Typical_performance": str, ... }, ... }
    label_to_text = {}
    for label_name, entry in knowledge.items():
        definition = entry.get('Definition', '')
        typical = entry.get('Typical_performance', '')
        text = (definition or '') + ' ' + (typical or '')
        label_to_text[label_name] = text.strip()

    probes = torch.zeros((len(label2id), hidden_size), dtype=torch.float32)
    # 反查id到label名称（label_map中通常为string索引）
    id_to_label = {idx: name for name, idx in label2id.items()}

    # 对每个标签编码
    # 只加载一次BERT
    from transformers import BertModel
    bert = BertModel.from_pretrained('./ZY-BERT')
    bert.eval()
    for idx in range(len(label2id)):
        label_name = id_to_label[idx]
        text = label_to_text.get(label_name, label_name)  # 若无知识条目，退化为标签名本身
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        with torch.no_grad():
            outputs = bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_vec = outputs.pooler_output.squeeze(0)  # [hidden]
            if cls_vec.shape[-1] != hidden_size:
                # 若隐藏维与期望不符，做简单线性投影到hidden_size
                proj = torch.nn.Linear(cls_vec.shape[-1], hidden_size, bias=False)
                cls_vec = proj(cls_vec)
            probes[idx] = cls_vec.float()

    return probes


def regenerate_assets(config, label2id: dict):
    """生成并保存 class_weights.pt 与 knowledge_probes.pt 到预处理目录。"""
    from transformers import AutoConfig, AutoTokenizer

    # 1) 计算类别权重
    class_weights = compute_class_weights_from_train(config.train_path, label2id)
    torch.save(class_weights, config.class_weights_path)

    # 2) 构建标签探针
    bert_cfg = AutoConfig.from_pretrained(config.bert_path)
    hidden_size = getattr(bert_cfg, 'hidden_size', 768)
    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
    knowledge_probes = build_knowledge_probes_from_knowledge(
        config.syndrome_knowledge_path,
        label2id,
        tokenizer,
        hidden_size,
    )
    torch.save(knowledge_probes, config.knowledge_probes_path)

    print(f"Saved class_weights -> {config.class_weights_path}")
    print(f"Saved knowledge_probes -> {config.knowledge_probes_path}")


def load_preprocessed_data():
    """加载所有预处理好的数据，返回字典"""
    import os
    
    # 路径配置
    base_dir = "preprocessed_data"
    
    # 加载标签映射
    with open(os.path.join(base_dir, "label_map.json"), 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    
    label2id = label_map['label2id']
    id2label = {int(k): v for k, v in label_map['id2label'].items()}
    
    # 加载距离矩阵
    distance_matrix = np.load(os.path.join(base_dir, "syndrome_distance_matrix.npy"))
    
    # 加载类别权重
    class_weights = torch.load(os.path.join(base_dir, "class_weights.pt"))
    
    # 加载知识探针
    knowledge_probes = torch.load(os.path.join(base_dir, "knowledge_probes.pt"))
    
    return {
        'label2id': label2id,
        'id2label': id2label,
        'distance_matrix': torch.from_numpy(distance_matrix),
        'class_weights': class_weights,
        'knowledge_probes': knowledge_probes
    }

