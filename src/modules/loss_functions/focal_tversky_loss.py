"""
Implementation of Focal Tversky Loss from https://github.com/Mr-TalhaIlyas/Loss-Functions-Package-Tensorflow-Keras-PyTorch
"""

import torch.nn.functional as F
from torch import nn

# Using alpha = beta = 0.5 and gamma = 1 for DiceLoss


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=2, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, outputs, targets):
        # comment out if your model contains a sigmoid or equivalent activation layer
        outputs = F.sigmoid(outputs)

        # flatten label and prediction tensors
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (outputs * targets).sum()
        FP = ((1 - targets) * outputs).sum()
        FN = (targets * (1 - outputs)).sum()

        Tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )
        FocalTversky = (1 - Tversky) ** self.gamma

        return FocalTversky
