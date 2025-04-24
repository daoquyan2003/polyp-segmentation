import torch
from monai.losses import GeneralizedDiceFocalLoss
from monai.utils import LossReduction
from torch import nn

from .lovasz_loss import LovaszLoss


class GeneralizedDiceFocalLovaszLoss(nn.Module):
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = True,
        softmax: bool = False,
        other_act=None,
        w_type: str = "square",
        reduction: LossReduction | str | None = "mean",
        smooth_nr: float = 1e-05,
        smooth_dr: float = 1e-05,
        batch: bool = False,
        gamma: float = 2.0,
        weight=None,
        lambda_gdl: float = 1.0,
        lambda_focal: float = 1.0,
        lambda_lovasz: float = 1.0,
    ):
        self.generalized_dice_focal_loss = GeneralizedDiceFocalLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            other_act=other_act,
            w_type=w_type,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
            gamma=gamma,
            weight=weight,
            lambda_gdl=lambda_gdl,
            lambda_focal=lambda_focal,
        )
        self.lambda_lovasz = lambda_lovasz
        self.lovasz_loss = LovaszLoss()

    def forward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if input.dim() != target.dim():
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} (nb dims: {len(input.shape)}) and {target.shape} (nb dims: {len(target.shape)}). "
                "if target is not one-hot encoded, please provide a tensor with shape B1H[WD]."
            )

        if target.shape[1] != 1 and target.shape[1] != input.shape[1]:
            raise ValueError(
                "number of channels for target is neither 1 (without one-hot encoding) nor the same as input, "
                f"got shape {input.shape} and {target.shape}."
            )

        gdl_focal = self.generalized_dice_focal_loss(input, target)
        lovasz = self.lovasz_loss(input, target)
        total_loss: torch.Tensor = self.lambda_lovasz * lovasz + gdl_focal
        return total_loss
