import torch
import torch.nn as nn
from monai.losses import DiceLoss


class LogCoshDiceLoss(nn.Module):
    def __init__(
        self,
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

    def forward(self, outputs, targets):
        return torch.log(
            (
                torch.exp(self.dice(outputs, targets))
                + torch.exp(-self.dice(outputs, targets))
            )
            / 2.0
        )
