import json
import os
import sys
from collections import defaultdict
from statistics import mean

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from monai.losses import (
    DiceLoss,
    GeneralizedDiceFocalLoss,
    GeneralizedDiceLoss,
)
from monai.metrics import (
    DiceMetric,
    GeneralizedDiceScore,
    MeanIoU,
    SSIMMetric,
    SurfaceDiceMetric,
)
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from .dataset import ColonDataset, my_collate
from .LinearWarmupCosine import LinearWarmupCosineAnnealingLR

image_folder = "./data/bkai-igh-neopolyp/train/train"
mask_folder = "./data/bkai-igh-neopolyp/train_gt/train_gt"
save_folder = "./output/bkai-igh-neopolyp/predict"
os.makedirs(save_folder, exist_ok=True)

image_path = []
mask_path = []

for root, dirs, files in os.walk(
    image_folder, topdown=False
):  # finds MRI files
    for name in files:
        if (
            name.endswith(".jpeg")
            or name.endswith(".jpg")
            or name.endswith(".png")
        ):
            apath = os.path.join(root, name)
            image_path.append(apath)

for root, dirs, files in os.walk(
    mask_folder, topdown=False
):  # finds MRI files
    for name in files:
        if (
            name.endswith(".jpeg")
            or name.endswith(".jpg")
            or name.endswith(".png")
        ):
            apath = os.path.join(root, name)
            mask_path.append(apath)

X_train, X_test, y_train, y_test = train_test_split(
    image_path, mask_path, test_size=0.2, random_state=42
)

sam_checkpoint = "./checkpoints/sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

transform = ResizeLongestSide(sam.image_encoder.img_size)

train_dataset = ColonDataset(X_train, y_train, sam.image_encoder.img_size)
train_dataloader = DataLoader(
    train_dataset, batch_size=2, shuffle=True, collate_fn=my_collate
)

val_dataset = ColonDataset(X_test, y_test, sam.image_encoder.img_size)
val_dataloader = DataLoader(
    val_dataset, batch_size=2, shuffle=True, collate_fn=my_collate
)


num_epochs = 2
# Set up the optimizer, hyperparameter tuning will improve performance here
lr = 4e-6
wd = 1e-4
torch.backends.cudnn.benchmark = True
parameters = (
    list(sam.mask_decoder.parameters())
    + list(sam.image_encoder.parameters())
    + list(sam.prompt_encoder.parameters())
)
scaler = torch.amp.GradScaler(device=device)
optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=wd)
scheduler = LinearWarmupCosineAnnealingLR(
    optimizer,
    warmup_epochs=30,
    max_epochs=num_epochs,
    warmup_start_lr=5e-7,
    eta_min=1e-6,
)

# Freeze something

for param in sam.prompt_encoder.parameters():
    param.requires_grad = False

for param in sam.image_encoder.parameters():
    param.requires_grad = False

root_dir = "./output/bkai-igh-neopolyp/SAM Finetune Enc Dec"

os.makedirs(root_dir, exist_ok=True)
writer = SummaryWriter(log_dir=os.path.join(root_dir, "runs"))
loss_fn = DiceLoss(to_onehot_y=False)

losses = []
dice_score = []
gd_score = []

best_dice = -1
best_gd = -1
best_score = -1

for epoch in range(num_epochs):
    epoch_losses = []

    for batch in train_dataloader:

        img, bbox, mask, gt_resized, original_image_size, input_size = (
            batch[0],
            batch[1],
            batch[2],
            batch[3],
            batch[4],
            batch[5],
        )

        batch_loss = 0

        for i in range(len(mask)):
            with torch.amp.autocast(device_type=device):

                image_embedding = sam.image_encoder(
                    img[i].unsqueeze(0).to(device)
                )

                orig_x, orig_y = (
                    original_image_size[i][0],
                    original_image_size[i][1],
                )
                col_x1, col_x2 = (
                    bbox[i][:, 1] * 1024 / orig_y,
                    bbox[i][:, 3] * 1024 / orig_y,
                )
                col_y1, col_y2 = (
                    bbox[i][:, 0] * 1024 / orig_x,
                    bbox[i][:, 2] * 1024 / orig_x,
                )

                box = np.array([col_x1, col_y1, col_x2, col_y2]).transpose()

                box_torch = torch.as_tensor(
                    box, dtype=torch.float, device=device
                )

                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    points=None,
                    boxes=box_torch,
                    masks=None,
                )

                low_res_masks, iou_predictions = sam.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                upscaled_masks = sam.postprocess_masks(
                    low_res_masks, input_size[i], original_image_size[i]
                ).to(device)

                binary_mask = torch.sigmoid(upscaled_masks)

                gt_binary_mask = mask[i].to(device)

                if binary_mask.size()[0] > 1:
                    binary_mask = torch.unsqueeze(
                        torch.sum(binary_mask, 0) / binary_mask.size()[0], 0
                    )

                loss = loss_fn(
                    binary_mask[0], gt_binary_mask.unsqueeze(0)
                ) / len(mask)
                scaler.scale(loss).backward()
                batch_loss += loss.item()

        scaler.step(optimizer)
        scaler.update()
        epoch_losses.append(batch_loss / len(mask))
        optimizer.zero_grad()
    scheduler.step()
    mean_epoch_loss = mean(epoch_losses)
    losses.append(mean_epoch_loss)
    # Log mean loss per epoch
    writer.add_scalar("Loss/train", mean_epoch_loss, epoch)
    print(f"EPOCH: {epoch}")
    print(f"Mean loss: {mean_epoch_loss}")

    with torch.no_grad():
        batch_dice = []
        batch_gd = []

        for batch in val_dataloader:

            img, bbox, mask, gt_resized, original_image_size, input_size = (
                batch[0],
                batch[1],
                batch[2],
                batch[3],
                batch[4],
                batch[5],
            )

            dice = DiceMetric()
            gd = GeneralizedDiceScore()

            for i in range(len(mask)):
                image_embedding = sam.image_encoder(
                    img[i].unsqueeze(0).to(device)
                )

                orig_x, orig_y = (
                    original_image_size[i][0],
                    original_image_size[i][1],
                )
                col_x1, col_x2 = (
                    bbox[i][:, 1] * 1024 / orig_y,
                    bbox[i][:, 3] * 1024 / orig_y,
                )
                col_y1, col_y2 = (
                    bbox[i][:, 0] * 1024 / orig_x,
                    bbox[i][:, 2] * 1024 / orig_x,
                )

                box = np.array([col_x1, col_y1, col_x2, col_y2]).transpose()

                num_masks = box.shape[0]
                box_torch = torch.as_tensor(
                    box, dtype=torch.float, device=device
                )
                sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                    points=None, boxes=box_torch, masks=None
                )

                low_res_masks, iou_predictions = sam.mask_decoder(
                    image_embeddings=image_embedding,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

                upscaled_masks = sam.postprocess_masks(
                    low_res_masks, input_size[i], original_image_size[i]
                )

                binary_mask = torch.sigmoid(upscaled_masks.detach().cpu())
                binary_mask = (binary_mask > 0.5).float()

                gt_binary_mask = mask[i].detach().cpu()

                if binary_mask.size()[0] > 1:
                    binary_mask = torch.unsqueeze(
                        torch.sum(binary_mask, 0) / binary_mask.size()[0], 0
                    )

                dice.reset()
                gd.reset()

                dice(binary_mask[0, :], gt_binary_mask.unsqueeze(0))
                gd(binary_mask[0, :], gt_binary_mask.unsqueeze(0))
                final_dice = dice.aggregate().numpy()[0]
                final_gd = gd.aggregate().numpy()[0]
                batch_dice.append(final_dice)
                batch_gd.append(final_gd)

        mean_dice = sum(batch_dice) / len(batch_dice)
        mean_gd = sum(batch_gd) / len(batch_gd)

        # Log mean validation Dice and GD scores per epoch
        writer.add_scalar("Dice/val", mean_dice, epoch)
        writer.add_scalar("GD/val", mean_gd, epoch)

        if (mean_dice) > best_dice:
            best_dice = mean_dice
            torch.save(
                sam.mask_decoder.state_dict(),
                os.path.join(root_dir, "dec_best_dice_model_DL.pth"),
            )
            torch.save(
                sam.image_encoder.state_dict(),
                os.path.join(root_dir, "img_enc_best_dice_model_DL.pth"),
            )
            torch.save(
                sam.prompt_encoder.state_dict(),
                os.path.join(root_dir, "prompt_enc_best_dice_model_DL.pth"),
            )

            print("saved new best dice model")

        if (mean_gd) > best_gd:
            best_gd = mean_gd
            torch.save(
                sam.mask_decoder.state_dict(),
                os.path.join(root_dir, "dec_best_GD_model_DL.pth"),
            )
            torch.save(
                sam.image_encoder.state_dict(),
                os.path.join(root_dir, "img_enc_best_GD_model_DL.pth"),
            )
            torch.save(
                sam.prompt_encoder.state_dict(),
                os.path.join(root_dir, "prompt_enc_best_GD_model_DL.pth"),
            )

            print("saved new best GD model")

        dice_score.append(mean_dice)
        gd_score.append(mean_gd)

    print(f"Mean val dice: {dice_score[-1]}")
    print(f"Mean val gd: {gd_score[-1]}")

model_path = root_dir

sam.prompt_encoder.load_state_dict(
    torch.load(
        os.path.join(model_path, "prompt_enc_best_dice_model_DL.pth")
    )  # nosec
)
sam.image_encoder.load_state_dict(
    torch.load(
        os.path.join(model_path, "img_enc_best_dice_model_DL.pth")
    )  # nosec
)
sam.mask_decoder.load_state_dict(
    torch.load(os.path.join(model_path, "dec_best_dice_model_DL.pth"))  # nosec
)
sam.eval()

with torch.no_grad():
    batch_dice = []
    batch_gd = []
    batch_iou = []

    for batch in val_dataloader:

        img, bbox, mask, gt_resized, original_image_size, input_size = (
            batch[0],
            batch[1],
            batch[2],
            batch[3],
            batch[4],
            batch[5],
        )

        dice = DiceMetric()
        gd = GeneralizedDiceScore()
        iou = MeanIoU()

        for i in range(len(mask)):
            image_embedding = sam.image_encoder(img[i].unsqueeze(0).to(device))

            orig_x, orig_y = (
                original_image_size[i][0],
                original_image_size[i][1],
            )
            col_x1, col_x2 = (
                bbox[i][:, 1] * 1024 / orig_y,
                bbox[i][:, 3] * 1024 / orig_y,
            )
            col_y1, col_y2 = (
                bbox[i][:, 0] * 1024 / orig_x,
                bbox[i][:, 2] * 1024 / orig_x,
            )

            box = np.array([col_x1, col_y1, col_x2, col_y2]).transpose()

            num_masks = box.shape[0]
            box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=None, boxes=box_torch, masks=None
            )

            low_res_masks, iou_predictions = sam.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            upscaled_masks = sam.postprocess_masks(
                low_res_masks, input_size[i], original_image_size[i]
            )

            binary_mask = torch.sigmoid(upscaled_masks.detach().cpu())
            binary_mask = (binary_mask > 0.5).float()

            gt_binary_mask = mask[i].detach().cpu()

            if binary_mask.size()[0] > 1:
                binary_mask = torch.unsqueeze(
                    torch.sum(binary_mask, 0) / binary_mask.size()[0], 0
                )

            dice.reset()
            gd.reset()
            iou.reset()

            dice(binary_mask[0, :], gt_binary_mask.unsqueeze(0))
            gd(binary_mask[0, :], gt_binary_mask.unsqueeze(0))
            iou(binary_mask[0, :], gt_binary_mask.unsqueeze(0))
            final_dice = dice.aggregate().numpy()[0]
            final_gd = gd.aggregate().numpy()[0]
            final_iou = iou.aggregate().numpy()[0]
            batch_dice.append(final_dice)
            batch_gd.append(final_gd)
            batch_iou.append(final_iou)

    m_dice = sum(batch_dice) / len(batch_dice)
    m_gd = sum(batch_gd) / len(batch_gd)
    m_iou = sum(batch_iou) / len(batch_iou)
    writer.add_scalar("Dice/val/final", m_dice)
    writer.add_scalar("GD/val/final", m_gd)
    writer.add_scalar("IoU/val/final", m_iou)
    print(f"Mean val dice: {m_dice}")
    print(f"Mean val gd: {m_gd}")
    print(f"Mean val iou: {m_iou}")

writer.close()
