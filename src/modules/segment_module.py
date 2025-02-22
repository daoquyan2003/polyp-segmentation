from typing import Any, List

import hydra
import torch
from omegaconf import DictConfig
from torch import nn
from torchmetrics import Dice, MeanMetric

from src.modules.components.lit_module import BaseLitModule
from src.modules.losses import load_loss
from src.modules.metrics import load_metrics


class SingleSegmentationLitModule(BaseLitModule):
    """Example of LightningModule for segmentation.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Model loop (model_step)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        network: DictConfig,
        optimizer: DictConfig,
        scheduler: DictConfig,
        logging: DictConfig,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """LightningModule with standalone train, val and test dataloaders.

        Args:
            network (DictConfig): Network config.
            optimizer (DictConfig): Optimizer config.
            scheduler (DictConfig): Scheduler config.
            logging (DictConfig): Logging config.
            args (Any): Additional arguments for pytorch_lightning.LightningModule.
            kwargs (Any): Additional keyword arguments for pytorch_lightning.LightningModule.
        """

        super().__init__(
            network, optimizer, scheduler, logging, *args, **kwargs
        )
        self.loss = load_loss(network.loss)

        main_metric, valid_metric_best, add_metrics = load_metrics(
            network.metrics
        )
        self.train_metric = main_metric.clone()
        self.train_add_metrics = add_metrics.clone(postfix="/train")
        self.valid_metric = main_metric.clone()
        self.valid_metric_best = valid_metric_best.clone()
        self.valid_add_metrics = add_metrics.clone(postfix="/valid")
        self.test_metric = main_metric.clone()
        self.test_add_metrics = add_metrics.clone(postfix="/test")

        self.train_dice = Dice(num_classes=3, average="macro")
        self.val_dice = Dice(num_classes=3, average="macro")
        self.test_dice = Dice(num_classes=3, average="macro")
        self.val_dice_best = valid_metric_best.clone()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.save_hyperparameters(logger=False)

    def model_step(self, batch: Any, *args: Any, **kwargs: Any) -> Any:
        images, masks = batch[0], batch[1]
        masks = masks.long()
        logits = self.forward(images)
        loss = self.loss(logits, masks)
        softmax = nn.Softmax(dim=1)
        preds = torch.argmax(softmax(logits), dim=1)
        return loss, preds, masks

    def on_train_start(self) -> None:
        # by default lightning executes validation step sanity checks before
        # training starts, so we need to make sure valid_metric_best doesn't store
        # accuracy from these checks
        self.valid_metric_best.reset()
        self.val_loss.reset()
        self.val_dice_best.reset()

    def training_step(self, batch: Any, batch_idx: int) -> Any:
        loss, preds, targets = self.model_step(batch, batch_idx)

        self.train_loss(loss)
        self.log(
            f"Mean_{self.loss.__class__.__name__}/train",
            self.train_loss,
            **self.logging_params,
        )

        self.train_metric(preds, targets)
        self.log(
            f"{self.train_metric.__class__.__name__}/train",
            self.train_metric,
            **self.logging_params,
        )

        self.train_dice(preds, targets.int())
        self.log(
            f"{self.train_dice.__class__.__name__}/train",
            self.train_dice,
            **self.logging_params,
        )

        self.train_add_metrics(preds, targets)
        self.log_dict(self.train_add_metrics, **self.logging_params)

        # Lightning keeps track of `training_step` outputs and metrics on GPU for
        # optimization purposes. This works well for medium size datasets, but
        # becomes an issue with larger ones. It might show up as a CPU memory leak
        # during training step. Keep it in mind.
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]) -> None:
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning
        # accumulates outputs from all batches of the epoch

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs
        pass

    def validation_step(self, batch: Any, batch_idx: int) -> Any:
        loss, preds, targets = self.model_step(batch, batch_idx)

        self.val_loss(loss)
        self.log(
            f"Mean_{self.loss.__class__.__name__}/valid",
            self.val_loss,
            **self.logging_params,
        )

        self.valid_metric(preds, targets)
        self.log(
            f"{self.valid_metric.__class__.__name__}/valid",
            self.valid_metric,
            **self.logging_params,
        )

        self.val_dice(preds, targets.int())
        self.log(
            f"{self.val_dice.__class__.__name__}/valid",
            self.val_dice,
            **self.logging_params,
        )

        self.valid_add_metrics(preds, targets)
        self.log_dict(self.valid_add_metrics, **self.logging_params)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        valid_metric = self.valid_metric.compute()  # get current valid metric
        dice_metric = self.val_dice.compute()
        self.valid_metric_best(valid_metric)  # update best so far valid metric
        self.val_dice_best(dice_metric)
        # log `valid_metric_best` as a value through `.compute()` method, instead
        # of as a metric object otherwise metric would be reset by lightning
        # after each epoch
        self.log(
            f"{self.valid_metric.__class__.__name__}/valid_best",
            self.valid_metric_best.compute(),
            **self.logging_params,
        )
        self.log(
            f"{self.val_dice.__class__.__name__}/valid_best",
            self.val_dice_best.compute(),
            **self.logging_params,
        )

    def test_step(self, batch: Any, batch_idx: int) -> Any:
        loss, preds, targets = self.model_step(batch, batch_idx)

        self.test_loss(loss)
        self.log(
            f"Mean_{self.loss.__class__.__name__}/test",
            self.test_loss,
            **self.logging_params,
        )

        self.test_metric(preds, targets)
        self.log(
            f"{self.test_metric.__class__.__name__}/test",
            self.test_metric,
            **self.logging_params,
        )

        self.test_dice(preds, targets.int())
        self.log(
            f"{self.test_dice.__class__.__name__}/test",
            self.test_dice,
            **self.logging_params,
        )

        self.test_add_metrics(preds, targets)
        self.log_dict(self.test_add_metrics, **self.logging_params)
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]) -> None:
        pass

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Any:
        images, masks = batch[0], batch[1]
        logits = self.forward(images)
        softmax = nn.Softmax(dim=1)
        preds = torch.argmax(softmax(logits), dim=1)
        outputs = {"logits": logits, "preds": preds, "targets": masks}
        return outputs
