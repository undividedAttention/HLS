import os
import networkx as nx
import json
import re
import numpy as np
import torch
from tqdm import tqdm
from itertools import combinations
from collections import Counter

def get_synonym_map():
    """定义同义词到标准词的映射。"""
    return {
        "风寒湿阻证": "风寒湿痹证", "风痰闭阻证": "风痰闭窍证", "热毒蕴结证": "热毒壅结证",
        "气虚血溢证": "气不摄血证", "气血不足证": "气血亏虚证", "肝肾亏损证": "肝肾亏虚证",
        "脾胃虚寒证": "脾胃阳虚证"
    }

def parse_syndrome_paths(filepath: str, synonym_map: dict) -> dict:
    """解析路径文件，并使用标准名称。"""
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
    print("正在计算类别权重...")
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    label_counts = Counter(item['merge_syndrome'] for item in train_data if item.get('merge_syndrome') in label2id)
    
    weights = torch.zeros(len(label2id))
    for label, count in label_counts.items():
        weights[label2id[label]] = count
    
    # 使用有效类别平衡 (Effective Number of Samples) 权重，更先进
    beta = 0.999
    effective_num = 1.0 - np.power(beta, weights.numpy())
    per_cls_weights = (1.0 - beta) / np.array(effective_num)
    per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(label2id)
    
    print("类别权重计算完成。")
    return torch.FloatTensor(per_cls_weights)


def preprocess_and_save(config):
    """V11.2 的预处理流程。"""
    print("--- 开始V11.2 (最终审查版) 预处理 ---")
    os.makedirs(config.preprocessed_data_dir, exist_ok=True)
    
    synonym_map = get_synonym_map()
    
    with open(config.vocab_path, 'r', encoding='utf-8') as f:
        all_labels = sorted([line.strip() for line in f if line.strip()])
    
    label2id = {label: i for i, label in enumerate(all_labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    class_weights = calculate_class_weights(config.train_path, label2id)
    torch.save(class_weights, config.class_weights_path)
    
    syndrome_paths = parse_syndrome_paths(config.syndrome_path_file, synonym_map)
    graph = build_knowledge_graph(syndrome_paths, all_labels)
    
    if not nx.is_directed_acyclic_graph(graph):
        print("警告: 图不是DAG，LCA可能不唯一。")

    try:
        depths = nx.shortest_path_length(graph, source="ROOT")
    except nx.NetworkXError as e:
        print(f"错误: 无法计算图深度，请检查图的连通性。 {e}")
        return

    distance_matrix = np.zeros((len(all_labels), len(all_labels)), dtype=np.float32)
    
    for i, j in tqdm(list(combinations(range(len(all_labels)), 2)), desc="计算距离矩阵"):
        label1, label2 = id2label[i], id2label[j]
        
        node1, node2 = label1, label2
        
        if node1 not in graph or node2 not in graph: continue
        
        try:
            lca = nx.lowest_common_ancestor(graph, node1, node2)
            if lca is None: continue
            
            dist = depths.get(node1, 0) + depths.get(node2, 0) - 2 * depths.get(lca, 0)
            distance_matrix[i, j] = distance_matrix[j, i] = dist
        except nx.NetworkXError:
            continue

    np.save(config.distance_matrix_path, distance_matrix)
    
    label_map = {'label2id': label2id, 'id2label': id2label}
    with open(config.label_map_path, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=4)
        
    print(f"--- V11.2 预处理完成 ---")

def load_preprocessed_data_v11(config):
    """加载V11.2预处理好的数据。"""
    with open(config.label_map_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    distance_matrix = np.load(config.distance_matrix_path + '.npy')
    
    # 如果class_weights文件不存在，创建默认的权重
    try:
        class_weights = torch.load(config.class_weights_path)
    except FileNotFoundError:
        print("class_weights.pt文件不存在，使用默认权重")
        class_weights = torch.ones(len(label_map['label2id']))
    
    return label_map['label2id'], label_map['id2label'], torch.from_numpy(distance_matrix), class_weights

