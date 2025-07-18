import torch
import torch.nn as nn          
import torch.nn.functional as F
import numpy as np  
import math

class CombinedLoss(nn.Module):
    def __init__(self, num_classes, lambda1=1.0, lambda2=0.15, tau=4.0):
        super(CombinedLoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.tau = tau
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        print(f"Modular loss function initialized with: λ1={self.lambda1}, λ2={self.lambda2}, τ={self.tau}")

    def forward(self, predictions, targets):
        total_loss = 0
        for p in predictions:
            logits_flat = p.permute(0, 2, 1).contiguous().view(-1, self.num_classes)
            targets_flat = targets.view(-1)
            
            min_len = min(logits_flat.size(0), targets_flat.size(0))
            logits_flat = logits_flat[:min_len]
            targets_flat = targets_flat[:min_len]
            
            loss_cel = self.ce_loss(logits_flat, targets_flat)

            if p.size(2) > 1:
                probs = F.softmax(p, dim=1).clamp(min=1e-8)
                log_probs = torch.log(probs)
                delta_log_probs = torch.abs(log_probs[:, :, 1:] - log_probs[:, :, :-1].detach())
                delta_sq = delta_log_probs.pow(2)
                loss_t_mse = torch.mean(torch.clamp(delta_sq, max=self.tau**2))
            else:
                loss_t_mse = 0.0

            total_loss += self.lambda1 * loss_cel + self.lambda2 * loss_t_mse
        return total_loss