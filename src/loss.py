import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridLoss(nn.Module):
    def __init__(self, distance_matrix, class_weights, device, 
                 lambda_focal=0.7, lambda_hls=0.3,
                 gamma=2.0, temperature_hls=1.0):
        super(HybridLoss, self).__init__()
        
        self.lambda_focal = lambda_focal
        self.lambda_hls = lambda_hls
        
        self.focal_loss_fn = self.FocalLoss(alpha=class_weights.to(device), gamma=gamma)
        
        self.register_buffer("distance_matrix", distance_matrix.to(device))
        self.temperature_hls = temperature_hls
        self.num_classes = distance_matrix.shape[0]

    class FocalLoss(nn.Module):
        def __init__(self, alpha, gamma=2.0, reduction='mean'):
            super().__init__()
            self.register_buffer('alpha', alpha)
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, logits, true_labels):
            ce_loss = F.cross_entropy(logits, true_labels, reduction='none')
            pt = torch.exp(-ce_loss)
            
            alpha_t = self.alpha[true_labels]
            
            focal_loss = alpha_t * (1 - pt)**self.gamma * ce_loss
            
            if self.reduction == 'mean': return focal_loss.mean()
            return focal_loss
            
    def hls_loss(self, logits, true_labels):
        soft_targets = F.softmax(-self.distance_matrix[true_labels] / self.temperature_hls, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        kl_div = F.kl_div(log_probs, soft_targets, reduction='batchmean')
        return kl_div

    def forward(self, logits, true_labels):
        loss_f = self.focal_loss_fn(logits, true_labels)
        loss_h = self.hls_loss(logits, true_labels)
        
        total_loss = self.lambda_focal * loss_f + self.lambda_hls * loss_h
        return total_loss

