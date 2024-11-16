import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.05, vocab_size=None, ignore_index=-100, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred, target):
        """
        pred (torch.Tensor): Predicted logits of shape [B, C] or [B, T, C]
        target (torch.Tensor): Target labels of shape [B] or [B, T]
        """
        # Handle different input shapes
        if pred.dim() == 3:
            pred = pred.view(-1, pred.size(-1))
            target = target.view(-1)
            
        # Create smoothed labels
        with torch.no_grad():
            label_smoothed = torch.zeros_like(pred)
            label_smoothed.fill_(self.smoothing / (self.vocab_size - 1))
            label_smoothed.scatter_(1, target.unsqueeze(1), self.confidence)
            
            # Handle padding or ignored indices
            if self.ignore_index >= 0:
                mask = (target == self.ignore_index)
                label_smoothed.masked_fill_(mask.unsqueeze(1), 0)
                
        # Calculate loss
        loss = torch.sum(-label_smoothed * F.log_softmax(pred, dim=-1), dim=-1)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss