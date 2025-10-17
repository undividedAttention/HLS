import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss实现，用于处理类别不平衡。"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, true_labels):
        ce_loss = F.cross_entropy(logits, true_labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class HybridLoss(nn.Module):
    """
    最终版混合损失：结合了Focal Loss和层次化标签相似性损失。
    L_total = alpha * L_Focal + (1 - alpha) * L_HLS
    """
    def __init__(self, distance_matrix, alpha=0.7, gamma=2.0, temperature=1.0, device='cpu'):
        super(HybridLoss, self).__init__()
        if not isinstance(distance_matrix, torch.Tensor):
            raise TypeError("distance_matrix必须是一个torch.Tensor")
        
        self.distance_matrix = distance_matrix.to(device)
        self.alpha = alpha
        self.temperature = temperature
        
        # 初始化Focal Loss作为硬约束
        self.focal_loss = FocalLoss(gamma=gamma, reduction='mean')

    def _create_soft_targets(self, true_labels):
        batch_distances = self.distance_matrix[true_labels]
        soft_targets = F.softmax(-batch_distances / self.temperature, dim=1)
        return soft_targets

    def forward(self, logits, true_labels):
        # 1. 计算Focal Loss (硬约束)
        loss_focal = self.focal_loss(logits, true_labels)

        # 2. 计算HLS损失 (软约束)
        log_probs = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            soft_targets = self._create_soft_targets(true_labels)
            
        loss_hls = F.kl_div(
            input=log_probs,
            target=soft_targets,
            reduction='batchmean'
        )
        
        # 3. 混合两种损失
        combined_loss = self.alpha * loss_focal + (1 - self.alpha) * loss_hls
        
        return combined_loss

