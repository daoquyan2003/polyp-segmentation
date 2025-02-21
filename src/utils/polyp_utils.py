import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]


def mask_overlay(
    image, mask, colors={1: (0, 255, 0), 2: (0, 0, 255)}, alpha=0.5
):
    """Overlay a segmentation mask on an image with transparency.

    Args:
        image (torch.Tensor or np.ndarray): Image (C, H, W) or (H, W, 3).
        mask (torch.Tensor or np.ndarray): Mask (H, W) with values {0, 1, 2}.
        colors (dict): Class index -> RGB color.
        alpha (float): Transparency of mask (0 = fully transparent, 1 = opaque).
    Returns:
        np.ndarray: Image with mask overlay.
    """
    mask = mask.squeeze()
    # Convert tensors to NumPy if needed
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    # Ensure image is (H, W, 3)
    if image.ndim == 3 and image.shape[0] in [1, 3]:
        image = np.transpose(
            image, (1, 2, 0)
        )  # Convert (C, H, W) -> (H, W, C)
    # Convert to uint8 if necessary
    if image.dtype != np.uint8:
        image = (image * 255).astype(
            np.uint8
        )  # Assuming image is normalized [0,1]
    # Create an overlay image
    overlay = image.copy()
    # Apply each class mask separately
    for class_id, color in colors.items():
        class_mask = (mask == class_id).astype(
            np.uint8
        )  # Binary mask (1, H, W)
        if np.any(class_mask):  # Only process if mask is present
            colored_mask = np.zeros_like(
                image, dtype=np.uint8
            )  # Empty (H, W, 3)
            for i in range(3):  # Assign RGB color
                colored_mask[:, :, i] = class_mask * color[i]
            # Blend mask with image where mask is present
            overlay = np.where(
                class_mask[:, :, None] > 0,
                (1 - alpha) * image + alpha * colored_mask,
                overlay,
            )
    return overlay.astype(np.uint8)


def imshow(img: np.ndarray, mask: np.ndarray, title: str = None) -> None:
    """Display the image with the mask overlay.

    Parameters:
    - img (np.ndarray): The image to be displayed.
    - mask (np.ndarray): The mask to overlay on the image.
    - title (str, optional): The title of the plot.
    """
    # Create the overlayed image with the mask
    overlayed_img = mask_overlay(img, mask)

    # Create a new figure with dynamic size
    fig, ax = plt.subplots(figsize=(6, 6))  # Adjust size as needed
    ax.imshow(overlayed_img)

    # Set title if provided
    if title:
        ax.set_title(title)

    ax.axis("off")  # Turn off the axis for better visual clarity
    plt.show()


def denormalize(x, mean=IMG_MEAN, std=IMG_STD) -> torch.Tensor:
    # 3, H, W, B
    ten = x.clone().permute(1, 2, 3, 0)
    for t, m, s in zip(ten, mean, std):
        t.mul_(s).add_(m)
    # B, 3, H, W
    return torch.clamp(ten, 0, 1).permute(3, 0, 1, 2)


def imshow_batch(images, masks, grid_shape=(8, 8)):
    images = denormalize(images)

    fig = plt.figure(figsize=(8, 8))

    for i, (mask, img) in enumerate(zip(masks, images)):
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        mask = mask.numpy().astype(np.uint8)

        ax = fig.add_subplot(
            grid_shape[0], grid_shape[1], i + 1, xticks=[], yticks=[]
        )
        ax.imshow(mask_overlay(img, mask))
    plt.show()
