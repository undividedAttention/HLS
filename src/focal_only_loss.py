import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalOnlyLoss(nn.Module):
    """
    消融实验2使用的损失函数：仅使用Focal Loss，不使用HLS Loss
    用于验证HLS正则化的作用
    """
    def __init__(self, class_weights, gamma=2.0, device='cpu'):
        super(FocalOnlyLoss, self).__init__()
        # 确保class_weights在正确的设备上
        if isinstance(class_weights, torch.Tensor):
            self.register_buffer('alpha', class_weights.to(device))
        else:
            self.register_buffer('alpha', torch.tensor(class_weights, device=device))
        self.gamma = gamma
    
    def forward(self, logits, true_labels):
        """
        Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        """
        ce_loss = F.cross_entropy(logits, true_labels, reduction='none')
        pt = torch.exp(-ce_loss)
        alpha_t = self.alpha[true_labels]
        focal_loss = alpha_t * (1 - pt)**self.gamma * ce_loss
        
        return focal_loss.mean()


