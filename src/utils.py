import os
import networkx as nx
import json
import re
import numpy as np
import torch
from tqdm import tqdm
from itertools import combinations

def get_synonym_mapping():
    """定义同义词到标准词的映射。"""
    return {
        "风寒湿阻证": "风寒湿痹证",
        "风痰闭阻证": "风痰闭窍证",
        "热毒蕴结证": "热毒壅结证",
        "气虚血溢证": "气不摄血证",
        "气血不足证": "气血亏虚证",
        "肝肾亏损证": "肝肾亏虚证",
        "脾胃虚寒证": "脾胃阳虚证"
    }

def parse_syndrome_paths(filepath: str, synonym_map: dict) -> dict:
    """解析路径文件，并使用标准名称。"""
    paths = {}
    reverse_synonym_map = {v: k for k, v in synonym_map.items()}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_syndrome = None
    for line in lines:
        line = line.strip()
        if not line: continue
        
        syndrome_match = re.match(r'症候:\s*(.*)', line)
        if syndrome_match:
            current_syndrome = syndrome_match.group(1).strip()
            # 转换为标准名称
            current_syndrome = synonym_map.get(current_syndrome, current_syndrome)
            continue
            
        path_match = re.match(r'路径:\s*(.*)', line)
        if path_match and current_syndrome:
            path_str = path_match.group(1).strip()
            path_list = [node.strip() for node in path_str.split('->')]
            # 将路径中的同义词也转换为标准名称
            standard_path = [synonym_map.get(node, node) for node in path_list]
            paths[current_syndrome] = standard_path
            current_syndrome = None
            
    # 处理孤立节点
    isolated_nodes = ["风痰瘀阻证", "正虚毒瘀证"]
    for node in isolated_nodes:
        node = synonym_map.get(node, node)
        if node not in paths:
            paths[node] = [node]
            
    return paths

def build_knowledge_graph(paths: dict, all_labels: list):
    """构建知识图谱，包含所有标签节点和中间节点。"""
    G = nx.DiGraph()
    ROOT_NODE = "ROOT"
    G.add_node(ROOT_NODE)

    all_nodes_in_paths = {ROOT_NODE}
    for path in paths.values():
        all_nodes_in_paths.update(path)
    
    # 确保图中包含词汇表里的所有标签节点
    all_graph_nodes = all_nodes_in_paths.union(set(all_labels))
    for node in all_graph_nodes:
        G.add_node(node)

    for syndrome, path in paths.items():
        if not path: continue
        G.add_edge(ROOT_NODE, path[0])
        for i in range(len(path) - 1):
            G.add_edge(path[i], path[i+1])

    # 将没有在路径中出现的孤立标签节点也连接到根节点
    for label in all_labels:
        if G.in_degree(label) == 0 and label != ROOT_NODE:
            G.add_edge(ROOT_NODE, label)
            
    return G

def get_adj_matrix(graph, node_map):
    """从图中提取归一化的邻接矩阵。"""
    num_nodes = len(node_map)
    adj = nx.to_numpy_array(graph, nodelist=list(node_map.keys()))
    # 添加自环
    adj += np.eye(num_nodes)
    # 对称归一化: D^{-1/2} A D^{-1/2}
    degree = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    
    normalized_adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return torch.from_numpy(normalized_adj).float()


def preprocess_and_save(config):
    """完整的预处理流程。"""
    print("--- 开始预处理 ---")
    os.makedirs(config.preprocessed_data_dir, exist_ok=True)
    
    synonym_map = get_synonym_mapping()
    
    with open(config.vocab_path, 'r', encoding='utf-8') as f:
        all_labels = sorted([line.strip() for line in f if line.strip()])
    
    syndrome_paths = parse_syndrome_paths(config.syndrome_path_file, synonym_map)
    
    # 构建包含所有节点的图
    graph = build_knowledge_graph(syndrome_paths, all_labels)
    # 包含所有图节点的映射
    all_graph_nodes = sorted(list(graph.nodes()))
    node2id = {node: i for i, node in enumerate(all_graph_nodes)}
    id2node = {i: node for node, i in node2id.items()}

    # 提取用于GCN的邻接矩阵
    adj_matrix = get_adj_matrix(graph, node2id)

    # 创建最终用于分类的标签映射
    label2id = {label: i for i, label in enumerate(all_labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    # 重新计算距离矩阵，基于包含所有节点的图
    depths = nx.shortest_path_length(graph, source="ROOT")
    distance_matrix = np.zeros((len(all_labels), len(all_labels)))
    
    for i, s1 in enumerate(all_labels):
        for j, s2 in enumerate(all_labels):
            if i == j: continue
            lca = nx.lowest_common_ancestor(graph, s1, s2)
            dist = depths.get(s1, 0) + depths.get(s2, 0) - 2 * depths.get(lca, 0)
            distance_matrix[i, j] = dist
            
    # --- 保存所有预处理文件 ---
    torch.save(torch.from_numpy(distance_matrix).float(), config.distance_matrix_path + ".pt")
    torch.save(adj_matrix, config.adj_matrix_path + ".pt")
    
    graph_map = {'node2id': node2id, 'id2node': id2node}
    with open(config.graph_map_path, 'w', encoding='utf-8') as f:
        json.dump(graph_map, f, ensure_ascii=False, indent=4)
        
    label_map = {'label2id': label2id, 'id2label': id2label}
    with open(config.label_map_path, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=4)
        
    print("--- 预处理完成 ---")


def load_preprocessed_data(config):
    """加载所有预处理好的数据。"""
    with open(config.label_map_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    with open(config.graph_map_path, 'r', encoding='utf-8') as f:
        graph_map = json.load(f)
        
    distance_matrix = torch.load(config.distance_matrix_path + ".pt")
    adj_matrix = torch.load(config.adj_matrix_path + ".pt")
    
    return label_map['label2id'], label_map['id2label'], graph_map, adj_matrix, distance_matrix

