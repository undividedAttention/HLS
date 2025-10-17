import networkx as nx
import json
import re
from itertools import combinations
from tqdm import tqdm
import os
import torch
import numpy as np

def get_synonym_mapping():
    """
    返回一个固定的同义词到标准词的映射字典。
    """
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
    """
    解析syndromeToPath.txt文件，并应用同义词映射。
    """
    paths = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_syndrome = None
    for line in lines:
        line = line.strip()
        if not line: continue
        
        syndrome_match = re.match(r'症候:\s*(.*)', line)
        if syndrome_match:
            syndrome_name = syndrome_match.group(1).strip()
            # 应用同义词映射，获取标准名称
            current_syndrome = synonym_map.get(syndrome_name, syndrome_name)
            continue
            
        path_match = re.match(r'路径:\s*(.*)', line)
        if path_match and current_syndrome:
            path_str = path_match.group(1).strip()
            path_list = [node.strip() for node in path_str.split('->')]
            # 使用标准名称作为键
            paths[current_syndrome] = path_list
            current_syndrome = None
            
    isolated_nodes = ["风痰瘀阻证", "正虚毒瘀证"]
    for node in isolated_nodes:
        # 同样应用同义词映射
        canonical_node = synonym_map.get(node, node)
        if canonical_node not in paths:
            paths[canonical_node] = [canonical_node]
            
    return paths

def build_knowledge_graph(paths: dict) -> nx.DiGraph:
    """
    根据解析的路径构建一个NetworkX有向图。
    """
    G = nx.DiGraph()
    ROOT_NODE = "ROOT"
    G.add_node(ROOT_NODE)
    
    for _, path in paths.items():
        if not path: continue
        G.add_edge(ROOT_NODE, path[0])
        for i in range(len(path) - 1):
            G.add_edge(path[i], path[i+1])
            
    if not nx.is_directed_acyclic_graph(G):
        print("警告: 构建的图包含环！")
    
    return G

def get_labels_from_vocab(vocab_path: str):
    """从vocabulary.txt文件中加载最终的标签列表。"""
    with open(vocab_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f if line.strip()]
    
    sorted_labels = sorted(list(set(labels))) # 去重并排序以保证一致性
    label2id = {label: i for i, label in enumerate(sorted_labels)}
    id2label = {i: label for i, label in enumerate(sorted_labels)}
    return sorted_labels, label2id, id2label

def preprocess_and_save(config):
    """
    执行所有预处理步骤并保存结果。
    """
    print("--- 开始数据预处理 ---")
    
    os.makedirs(config.preprocessed_data_dir, exist_ok=True)
    
    # 1. 加载同义词映射和从词汇表获取标准标签
    synonym_map = get_synonym_mapping()
    labels, label2id, id2label = get_labels_from_vocab(config.vocab_path)
    with open(config.label_map_path, 'w', encoding='utf-8') as f:
        json.dump({'label2id': label2id, 'id2label': id2label}, f, ensure_ascii=False, indent=4)
    print(f"标签映射已从 {config.vocab_path} 加载并保存至: {config.label_map_path}")

    # 2. 解析路径文件（应用同义词映射）并构建知识图谱
    syndrome_paths = parse_syndrome_paths(config.syndrome_path_file, synonym_map)
    graph = build_knowledge_graph(syndrome_paths)
    nx.write_gml(graph, config.graph_path)
    print(f"知识图谱已保存至: {config.graph_path}")
    
    # 3. 计算距离矩阵
    depths = nx.shortest_path_length(graph, source="ROOT")
    num_labels = len(labels)
    distance_matrix = np.zeros((num_labels, num_labels), dtype=np.float32)

    print("正在计算层次距离矩阵...")
    for i in tqdm(range(num_labels), desc="计算距离"):
        for j in range(i, num_labels):
            s1 = id2label[i]
            s2 = id2label[j]

            if i == j:
                dist = 0
            else:
                # 确保使用在路径文件中存在的症候来查找路径
                node1_path = syndrome_paths.get(s1)
                node2_path = syndrome_paths.get(s2)

                # 如果标签没有对应的路径（可能是不在146个标准中的），则给予一个最大惩罚距离
                if not node1_path or not node2_path:
                    dist = 2 * (max(depths.values())) 
                else:
                    node1 = node1_path[-1]
                    node2 = node2_path[-1]
                    lca = nx.lowest_common_ancestor(graph, node1, node2)
                    dist = depths[node1] + depths[node2] - 2 * depths[lca]

            distance_matrix[i, j] = distance_matrix[j, i] = dist

    np.save(config.distance_matrix_path, distance_matrix)
    print(f"距离矩阵 (shape: {distance_matrix.shape}) 已保存至: {config.distance_matrix_path}.npy")
    print("--- 数据预处理完成 ---")

def load_preprocessed_data(config):
    """加载所有预处理好的数据。"""
    with open(config.label_map_path, 'r', encoding='utf-8') as f:
        label_maps = json.load(f)
    
    distance_matrix_path = config.distance_matrix_path + '.npy'
    if not os.path.exists(distance_matrix_path):
        raise FileNotFoundError(f"错误: 预处理文件 {distance_matrix_path} 不存在。请先运行带有 --do_preprocess 参数的命令。")
        
    distance_matrix = np.load(distance_matrix_path)
    
    return label_maps['label2id'], label_maps['id2label'], torch.from_numpy(distance_matrix)

