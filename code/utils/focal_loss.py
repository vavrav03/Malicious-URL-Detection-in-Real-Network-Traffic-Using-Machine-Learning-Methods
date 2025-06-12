import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=-1, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        print("Setting focal loss", alpha, gamma, reduction)

    def forward(self, logits, targets):
        if logits.dim() == 2 and logits.size(1) == 2:
            logits = logits[:, 1]
        targets = targets.float()
        return sigmoid_focal_loss(logits, targets, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)