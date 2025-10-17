import torch
import torch.nn as nn
import torch.nn.functional as F

class HLSLoss(nn.Module):
    """
    层次化标签相似性损失函数 (Hierarchical Label Similarity Loss)
    """
    def __init__(self, distance_matrix, temperature=1.0, device='cpu'):
        super(HLSLoss, self).__init__()
        if not isinstance(distance_matrix, torch.Tensor):
            raise TypeError("distance_matrix必须是一个torch.Tensor")
        
        self.distance_matrix = distance_matrix.to(device)
        self.temperature = temperature
        self.num_classes = distance_matrix.shape[0]

    def _create_soft_targets(self, true_labels):
        """
        根据真实标签和距离矩阵构建软目标分布。
        """
        # 从距离矩阵中高效地索引出每个真实标签对应的距离向量
        batch_distances = self.distance_matrix[true_labels]
        soft_targets = F.softmax(-batch_distances / self.temperature, dim=1)
        return soft_targets

    def forward(self, logits, true_labels):
        """
        计算HLS损失。
        """
        log_probs = F.log_softmax(logits, dim=1)
        
        with torch.no_grad():
            soft_targets = self._create_soft_targets(true_labels)
            
        loss = F.kl_div(
            input=log_probs,
            target=soft_targets,
            reduction='batchmean'
        )
        
        return loss

