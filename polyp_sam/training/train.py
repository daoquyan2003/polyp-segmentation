import os
from statistics import mean

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ..utils.metrics import DiceMetric, GeneralizedDiceScore


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        scaler,
        device,
        log_dir,
        model_weight_path,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.scaler = scaler
        self.device = device
        self.writer = SummaryWriter(log_dir)
        self.best_dice = -1
        self.best_gd = -1
        self.best_score = -1
        self.model_weight_path = model_weight_path

    def train_epoch(self, epoch):
        # self.model.train()
        epoch_losses = []
        for batch in tqdm(self.train_loader):
            (
                images,
                bboxes,
                masks,
                gt_resized,
                original_image_size,
                input_size,
            ) = batch
            images = images.to(self.device)
            masks = masks.to(self.device)
            batch_loss = 0
            for i in range(len(masks)):
                with torch.amp.autocast(device_type=self.device):
                    image_embedding = self.model.image_encoder(
                        images[i].unsqueeze(0).to(self.device)
                    )

                    orig_x, orig_y = (
                        original_image_size[i][0],
                        original_image_size[i][1],
                    )
                    col_x1, col_x2 = (
                        bboxes[i][:, 1] * 1024 / orig_y,
                        bboxes[i][:, 3] * 1024 / orig_y,
                    )
                    col_y1, col_y2 = (
                        bboxes[i][:, 0] * 1024 / orig_x,
                        bboxes[i][:, 2] * 1024 / orig_x,
                    )

                    box = np.array(
                        [col_x1, col_y1, col_x2, col_y2]
                    ).transpose()

                    box_torch = torch.as_tensor(
                        box, dtype=torch.float, device=self.device
                    )

                    sparse_embeddings, dense_embeddings = (
                        self.model.prompt_encoder(
                            points=None,
                            boxes=box_torch,
                            masks=None,
                        )
                    )

                    low_res_masks, iou_predictions = self.model.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=self.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )

                    upscaled_masks = self.model.postprocess_masks(
                        low_res_masks, input_size[i], original_image_size[i]
                    ).to(self.device)

                    binary_mask = torch.sigmoid(upscaled_masks)

                    gt_binary_mask = masks[i].to(self.device)

                    if binary_mask.size()[0] > 1:
                        binary_mask = torch.unsqueeze(
                            torch.sum(binary_mask, 0) / binary_mask.size()[0],
                            0,
                        )

                    loss = self.loss_fn(
                        binary_mask[0], gt_binary_mask.unsqueeze(0)
                    ) / len(masks)
                    self.scaler.scale(loss).backward()
                    batch_loss += loss.item()

            self.scaler.step(self.optimizer)
            self.scaler.update()
            epoch_losses.append(batch_loss / len(masks))
            self.optimizer.zero_grad()

        self.scheduler.step()
        avg_loss = mean(epoch_losses)
        self.writer.add_scalar("Loss/train", avg_loss, epoch)
        return avg_loss

    def validate(self, epoch):
        self.model.eval()
        dice_metric = DiceMetric()
        gd_metric = GeneralizedDiceScore()
        val_loss = 0

        with torch.no_grad():
            batch_dice = []
            batch_gd = []

            for batch in tqdm(self.val_loader):
                (
                    images,
                    bboxes,
                    masks,
                    gt_resized,
                    original_image_size,
                    input_size,
                ) = batch
                images = images.to(self.device)
                masks = masks.to(self.device)

                for i in range(len(masks)):
                    image_embedding = self.model.image_encoder(
                        images[i].unsqueeze(0).to(self.device)
                    )

                    orig_x, orig_y = (
                        original_image_size[i][0],
                        original_image_size[i][1],
                    )
                    col_x1, col_x2 = (
                        bboxes[i][:, 1] * 1024 / orig_y,
                        bboxes[i][:, 3] * 1024 / orig_y,
                    )
                    col_y1, col_y2 = (
                        bboxes[i][:, 0] * 1024 / orig_x,
                        bboxes[i][:, 2] * 1024 / orig_x,
                    )

                    box = np.array(
                        [col_x1, col_y1, col_x2, col_y2]
                    ).transpose()

                    num_masks = box.shape[0]
                    box_torch = torch.as_tensor(
                        box, dtype=torch.float, device=self.device
                    )
                    sparse_embeddings, dense_embeddings = (
                        self.model.prompt_encoder(
                            points=None, boxes=box_torch, masks=None
                        )
                    )

                    low_res_masks, iou_predictions = self.model.mask_decoder(
                        image_embeddings=image_embedding,
                        image_pe=self.model.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                    )

                    upscaled_masks = self.model.postprocess_masks(
                        low_res_masks, input_size[i], original_image_size[i]
                    )

                    binary_mask = torch.sigmoid(upscaled_masks.detach().cpu())
                    binary_mask = (binary_mask > 0.5).float()

                    gt_binary_mask = masks[i].detach().cpu()

                    if binary_mask.size()[0] > 1:
                        binary_mask = torch.unsqueeze(
                            torch.sum(binary_mask, 0) / binary_mask.size()[0],
                            0,
                        )

                    dice_metric.reset()
                    gd_metric.reset()

                    dice_metric(binary_mask[0, :], gt_binary_mask.unsqueeze(0))
                    gd_metric(binary_mask[0, :], gt_binary_mask.unsqueeze(0))
                    final_dice = dice_metric.aggregate().numpy()[0]
                    final_gd = gd_metric.aggregate().numpy()[0]
                    batch_dice.append(final_dice)
                    batch_gd.append(final_gd)

            if (sum(batch_dice) / len(batch_dice)) > self.best_dice:
                self.best_dice = sum(batch_dice) / len(batch_dice)
                torch.save(
                    self.model.mask_decoder.state_dict(),
                    os.path.join(
                        self.model_weight_path, "dec_best_dice_model_DL.pth"
                    ),
                )
                torch.save(
                    self.model.image_encoder.state_dict(),
                    os.path.join(
                        self.model_weight_path,
                        "img_enc_best_dice_model_DL.pth",
                    ),
                )
                torch.save(
                    self.model.prompt_encoder.state_dict(),
                    os.path.join(
                        self.model_weight_path,
                        "prompt_enc_best_dice_model_DL.pth",
                    ),
                )

                print("saved new best dice model")

            if (sum(batch_gd) / len(batch_gd)) > self.best_gd:
                self.best_gd = sum(batch_gd) / len(batch_gd)
                torch.save(
                    self.model.mask_decoder.state_dict(),
                    os.path.join(
                        self.model_weight_path, "dec_best_GD_model_DL.pth"
                    ),
                )
                torch.save(
                    self.model.image_encoder.state_dict(),
                    os.path.join(
                        self.model_weight_path, "img_enc_best_GD_model_DL.pth"
                    ),
                )
                torch.save(
                    self.model.prompt_encoder.state_dict(),
                    os.path.join(
                        self.model_weight_path,
                        "prompt_enc_best_GD_model_DL.pth",
                    ),
                )

                print("saved new best GD model")

        dice_score = sum(batch_dice) / len(batch_dice)
        gd_score = sum(batch_gd) / len(batch_gd)
        self.writer.add_scalar("Metrics/Dice", dice_score, epoch)
        self.writer.add_scalar("Metrics/GeneralizedDice", gd_score, epoch)

        return dice_score, gd_score

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            dice_score, gd_score = self.validate(epoch)
            print(
                f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Dice: {dice_score}, GD: {gd_score}"
            )
