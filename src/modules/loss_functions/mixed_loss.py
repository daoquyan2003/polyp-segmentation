import torch
from monai.losses import DiceLoss
from torch import nn
from torchvision.ops import sigmoid_focal_loss


class MixedLoss(nn.Module):
    """Implementation based on https://www.kaggle.com/code/iafoss/unet34-dice-0-87."""

    def __init__(
        self,
        alpha=0.25,
        gamma=2,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = True,
        softmax: bool = False,
        other_act=None,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-05,
        smooth_dr: float = 1e-05,
        batch: bool = False,
        weight=None,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
            weight=weight,
        )

    def forward(self, input, target):
        loss = sigmoid_focal_loss(
            inputs=input,
            targets=target,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction="mean",
        ) - torch.log(1 - self.dice(input, target))
        return loss
