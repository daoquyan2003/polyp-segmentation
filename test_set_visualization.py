import os

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.datamodules.components.neopolyp_dataset_v1 import NeoPolypDataset
from src.modules.models.unet_versions.UNet import UNet
from src.utils.polyp_utils import denormalize


def create_green_overlay(mask):
    green = np.zeros((*mask.shape, 3), dtype=np.float32)
    green[..., 1] = 1.0  # green channel
    return green


transforms = A.Compose(
    [
        A.Resize(384, 384),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        A.ToTensorV2(),
    ]
)

dataset = NeoPolypDataset(
    data_dir="data",
    transforms=transforms,
    mode="test",
)

device = "cuda" if torch.cuda.is_available() else "cpu"

weight_path = "..."
model = UNet(in_channels=3, n_classes=1)
model.load_state_dict(torch.load(weight_path))  # nosec
model.to(device)
model.eval()

output_dir = "output/visualization/NeoPolyp"
os.makedirs(output_dir, exist_ok=True)

for i in range(dataset.__len__()):
    image, mask = dataset[i]
    image_name = dataset.image_paths[i]
    save_path = os.path.join(output_dir, f"{os.path.basename(image_name)}.png")
    image = image.to(device).unsqueeze(0)
    pred = model(image)
    pred = torch.sigmoid(pred)
    pred = pred >= 0.5
    pred = pred.squeeze().detach().cpu().numpy()
    image = denormalize(image).squeeze().detach().cpu().numpy()
    mask = mask.squeeze().cpu().numpy()
    if image.shape[0] == 3 and image.ndim == 3:
        image = np.transpose(image, (1, 2, 0))  # From (C, H, W) to (H, W, C)

    # Plot with transparent overlay
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.suptitle(f"Image: {image_name}", fontsize=12)

    # Ground truth overlay
    axes[0].imshow(image)
    axes[0].imshow(
        create_green_overlay(mask),
        alpha=(mask * 0.4),  # Only 1s get 40% opacity
    )
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    # Prediction overlay
    axes[1].imshow(image)
    axes[1].imshow(
        create_green_overlay(pred),
        alpha=(pred * 0.4),  # Only 1s get 40% opacity
    )
    axes[1].set_title("Model Prediction")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
