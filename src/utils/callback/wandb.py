import os
from typing import Any

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
from pytorch_lightning.callbacks import Callback
from torchvision.utils import make_grid

from src.utils.polyp_utils import denormalize, mask_overlay


class WandbCallback(Callback):
    def __init__(
        self,
        image_path: str = "data/train/train/0a22abd004c33abf3ae2136cd9dd77ae.jpeg",
        # data_path: str = "data",
        n_images_to_log: int = 5,
        img_size: int = 512,
    ):
        self.img_size = img_size
        self.n_images_to_log = (
            n_images_to_log  # number of logged images when eval
        )

        self.four_first_preds = []
        self.four_first_targets = []
        self.four_first_batch = []
        self.four_first_image = []
        self.show_pred = []
        self.show_target = []
        self.four_fisrt_paths = []

        self.batch_size = 1
        self.num_samples = 8
        self.num_batch = 0

        mask_path = image_path.replace("train", "train_gt")

        self.sample_image = np.array(Image.open(image_path).convert("RGB"))
        self.sample_image_height, self.sample_image_width = (
            self.sample_image.shape[0],
            self.sample_image.shape[1],
        )
        mask = np.asarray(Image.open(mask_path).convert("RGB"))
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        self.sample_mask = np.zeros_like(mask)  # Default to 0
        self.sample_mask[(mask >= 50) & (mask <= 100)] = 1
        self.sample_mask[mask > 100] = 2

        self.transform = Compose(
            [
                A.Resize(self.img_size, self.img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2(),
            ]
        )

    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        wandb_logger = trainer.logger
        wandb_logger.log_image(
            key="Real Mask",
            images=[
                Image.fromarray(
                    mask_overlay(self.sample_image, self.sample_mask)
                )
            ],
        )

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        transformed = self.transform(image=self.sample_image)
        image = transformed["image"]  # (3, img_size, img_size)
        image = image.unsqueeze(0).to(
            trainer.model.device
        )  # (1, 3, img_size, img_size)

        pred_mask = trainer.model(image)
        pred_mask = pred_mask.detach()  # (1, 3, img_size, img_size)
        softmax = torch.nn.Softmax(dim=1)
        pred_mask = torch.argmax(
            softmax(pred_mask), dim=1
        )  # (1, img_size, img_size)
        pred_mask = pred_mask.permute(1, 2, 0)
        pred_mask = pred_mask.cpu().numpy().astype(np.uint8)
        pred_mask = cv2.resize(
            pred_mask,
            (self.sample_image_width, self.sample_image_height),
            interpolation=cv2.INTER_CUBIC,
        )

        wandb_logger = trainer.logger
        wandb_logger.log_image(
            key="predicted mask",
            images=[
                Image.fromarray(mask_overlay(self.sample_image, pred_mask))
            ],
        )

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        preds = outputs["preds"]
        targets = outputs["targets"]
        self.batch_size = preds.shape[0]
        if len(self.four_first_batch) >= self.batch_size:
            return
        # Store predictions, targets, and images together
        for i in range(self.batch_size):
            if len(self.four_first_batch) >= self.batch_size:
                break
            self.four_first_batch.append(
                {
                    "image": batch[0][
                        i
                    ],  # Assuming batch[0] is the image tensor
                    "pred": preds[i],
                    "target": targets[i],
                }
            )

    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ):
        for i, batch_data in enumerate(self.four_first_batch):
            image = batch_data["image"]
            pred = batch_data["pred"]
            target = batch_data["target"]

            # Process image
            image = image.unsqueeze(0)
            image = denormalize(image)
            image = image.squeeze()  # (3, 768, 768)
            image = image.cpu().numpy()
            image = (image * 255).astype(np.uint8)
            image = np.transpose(image, (1, 2, 0))

            # Process prediction
            pred = pred.unsqueeze(0)
            pred = pred.cpu().numpy().astype(np.uint8)
            log_pred = mask_overlay(image, pred)
            log_pred = np.transpose(log_pred, (2, 0, 1))
            log_pred = torch.from_numpy(log_pred)
            self.show_pred.append(log_pred)

            # Process target
            target = target.unsqueeze(0)
            target = target.cpu().numpy().astype(np.uint8)
            log_target = mask_overlay(image, target)
            log_target = np.transpose(log_target, (2, 0, 1))
            log_target = torch.from_numpy(log_target)
            self.show_target.append(log_target)

        # Create grids and log to wandb
        stack_pred = torch.stack(self.show_pred)
        stack_target = torch.stack(self.show_target)

        grid_pred = make_grid(stack_pred, nrow=4)
        grid_target = make_grid(stack_target, nrow=4)

        grid_pred_np = grid_pred.numpy().transpose(1, 2, 0)
        grid_target_np = grid_target.numpy().transpose(1, 2, 0)

        grid_pred_np = Image.fromarray(grid_pred_np)
        grid_target_np = Image.fromarray(grid_target_np)

        wandb_logger = trainer.logger
        wandb_logger.log_image(
            key="Prediction vs Ground Truth",
            images=[grid_pred_np, grid_target_np],
        )

        # Clear lists for the next epoch
        self.four_first_preds.clear()
        self.four_first_targets.clear()
        self.four_first_batch.clear()
        self.show_pred.clear()
        self.show_target.clear()

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.n_images_to_log <= 0:
            return

        logger = trainer.logger

        preds = outputs["preds"]
        targets = outputs["targets"]
        images, ys, image_paths = batch

        images = denormalize(images)
        for img, pred, target, id in zip(images, preds, targets, image_paths):
            if self.n_images_to_log <= 0:
                break

            img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pred = pred.cpu().numpy().astype(np.uint8)
            target = target.cpu().numpy().astype(np.uint8)

            log_pred = mask_overlay(img, pred)
            log_target = mask_overlay(img, target)

            log_img = Image.fromarray(img)
            log_pred = Image.fromarray(log_pred)
            log_target = Image.fromarray(log_target)

            logger.log_image(
                key="Sample",
                images=[log_img, log_pred, log_target],
                caption=[id + "-Real", id + "-Predict", id + "-GroundTruth"],
            )

            self.n_images_to_log -= 1
