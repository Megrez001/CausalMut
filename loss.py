import torch
import torch.nn as nn

class FocalIPSLoss(nn.Module):
    def __init__(self, alpha, beta, gamma):
        super(FocalIPSLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def forward(self, inputs, targets, ips_weights=None):
        squared_error = (inputs - targets) ** 2
        abs_error = torch.abs(inputs - targets)
        
        ips_weight_term = self.alpha * ips_weights
        focal_weights = (abs_error + 1) ** self.gamma
        focal_weight_term = self.beta * focal_weights
        
        combined_weights = (1 + ips_weight_term) * (1 + focal_weight_term)
        weighted_loss = combined_weights * squared_error
        
        return weighted_loss.mean()