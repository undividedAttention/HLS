import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from transformers import BertModel
import math

class GraphConvolutionLayer(nn.Module):
    """简化的图卷积层实现。"""
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GraphAwareClassifier(nn.Module):
    """使用GCN生成层次感知分类器权重的模块。"""
    def __init__(self, adj_matrix, all_graph_nodes_map, target_labels_map, in_features, gcn_hidden, num_gcn_layers=2):
        super(GraphAwareClassifier, self).__init__()
        self.adj_matrix = adj_matrix
        self.node2id = all_graph_nodes_map['node2id']
        self.label2id = target_labels_map
        self.num_all_nodes = len(self.node2id)
        
        # 初始的节点嵌入，这是可学习的
        self.node_embeddings = nn.Embedding(self.num_all_nodes, gcn_hidden)

        # GCN层
        self.gcn_layers = nn.ModuleList()
        for _ in range(num_gcn_layers):
            self.gcn_layers.append(GraphConvolutionLayer(gcn_hidden, gcn_hidden))
        
        # 将GCN输出映射到BERT的隐藏层维度，以进行点积
        self.map_to_bert_hidden = nn.Linear(gcn_hidden, in_features)

        # 创建一个索引，用于从所有节点的嵌入中挑选出目标标签的嵌入
        target_indices = [self.node2id[label] for label in self.label2id.keys()]
        self.register_buffer('target_indices', torch.LongTensor(target_indices))

    def forward(self, text_embedding):
        # 1. 通过GCN传播，得到所有节点的层次感知嵌入
        x = self.node_embeddings.weight
        for layer in self.gcn_layers:
            x = F.relu(layer(x, self.adj_matrix))
        
        # 2. 将GCN输出映射到与文本嵌入相同的维度
        hierarchical_embeddings = self.map_to_bert_hidden(x)
        
        # 3. 从所有节点嵌入中，挑选出目标分类标签对应的嵌入作为分类器权重
        classifier_weights = hierarchical_embeddings[self.target_indices]
        
        # 4. 计算logits (batch_size, hidden_size) * (hidden_size, num_classes)
        logits = torch.matmul(text_embedding, classifier_weights.t())
        
        return logits


class SyndromeClassifier(nn.Module):
    def __init__(self, bert_path, adj_matrix, all_graph_nodes_map, target_labels_map):
        super(SyndromeClassifier, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_path)
        self.bert_config = self.bert.config
        bert_hidden_size = self.bert_config.hidden_size
        
        # 使用新的图感知分类器
        self.classifier = GraphAwareClassifier(
            adj_matrix=adj_matrix,
            all_graph_nodes_map=all_graph_nodes_map,
            target_labels_map=target_labels_map,
            in_features=bert_hidden_size,
            gcn_hidden=256 # GCN的隐藏维度可以自行调整
        )
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits

