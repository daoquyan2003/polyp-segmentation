import glob
import io
import os
from pathlib import Path
from typing import Any, Callable, Literal

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class KvasirSEGDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        transforms: Callable | None = None,
        read_mode: str = "pillow",
        train_val_test_ratio: list | tuple = (0.8, 0.1, 0.1),
        mode: Literal["train", "val", "test"] = "train",
    ) -> None:
        self.data_dir = data_dir
        self.transforms = transforms
        self.read_mode = read_mode
        self.train_val_test_ratio = train_val_test_ratio

        assert (
            len(self.train_val_test_ratio) == 3
        ), "train_val_test_ratio must have length of 3."

        assert (
            sum(self.train_val_test_ratio) == 1
        ), "train_val_test_ratio must sum up to 1."

        data_path = os.path.join(self.data_dir, "Kvasir-SEG")

        image_dir = os.path.join(data_path, "images")
        mask_dir = os.path.join(data_path, "masks")

        image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.jpg")))

        X_train, X_val_test, y_train, y_val_test = train_test_split(
            image_paths,
            mask_paths,
            test_size=(1 - self.train_val_test_ratio[0]),
            random_state=42,
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_val_test,
            y_val_test,
            test_size=(
                self.train_val_test_ratio[2]
                / (1 - self.train_val_test_ratio[0])
            ),
            random_state=42,
        )

        if mode == "train":
            self.image_paths = X_train
            self.mask_paths = y_train
        elif mode == "val":
            self.image_paths = X_val
            self.mask_paths = y_val
        else:
            self.image_paths = X_test
            self.mask_paths = y_test

        assert len(self.image_paths) == len(
            self.mask_paths
        ), "Number of images and masks must be the same."

    def _read_image_(self, image: Any) -> np.ndarray:
        """Read image from source.

        Args:
            image (Any): Image source. Could be str, Path or bytes.

        Returns:
            np.ndarray: Loaded image.
        """

        if self.read_mode == "pillow":
            if not isinstance(image, (str, Path)):
                image = io.BytesIO(image)
            image = np.asarray(Image.open(image).convert("RGB"))
        elif self.read_mode == "cv2":
            if not isinstance(image, (str, Path)):
                image = np.frombuffer(image, np.uint8)
                image = cv2.imdecode(image, cv2.COLOR_RGB2BGR)
            else:
                image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise NotImplementedError("read_mode must be `pillow` or `cv2`.")
        return image

    def _read_mask_(self, mask: Any) -> np.ndarray:
        """Read mask from source.

        Args:
            mask (Any): Mask source. Could be str, Path or bytes.

        Returns:
            np.ndarray: Loaded mask.

        Notes:
            The mask is read as RGB image and then converted to grayscale.
            The grayscale values are then thresholded to create a binary mask
            where 0 represents background. 1 and 2 represent 2 segmentation classes.
        """
        if self.read_mode == "pillow":
            if not isinstance(mask, (str, Path)):
                mask = io.BytesIO(mask)
            mask = np.asarray(Image.open(mask).convert("RGB"))
        elif self.read_mode == "cv2":
            if not isinstance(mask, (str, Path)):
                mask = np.frombuffer(mask, np.uint8)
                mask = cv2.imdecode(mask, cv2.COLOR_RGB2BGR)
            else:
                mask = cv2.imread(mask)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        else:
            raise NotImplementedError("read_mode must be `pillow` or `cv2`.")
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        processed_mask = np.zeros_like(mask)  # Default to 0
        processed_mask[mask > 0] = 1

        return processed_mask

    def _process_image_mask_(
        self, image: np.ndarray, mask: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process image and mask, including transforms, etc.

        Args:
            image (np.ndarray): Image in np.ndarray format.
            mask (np.ndarray): Mask in np.ndarray format.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Processed image and mask.
        """
        if self.transforms:
            result = self.transforms(image=image, mask=mask, dummy_labels=[0])
            if isinstance(result["image"], np.ndarray):
                if result["image"].ndim == 3 and result["image"].shape[2] == 3:
                    return (
                        torch.from_numpy(result["image"]).permute(2, 0, 1),
                        torch.from_numpy(result["mask"]).to(torch.float32),
                    )
                else:
                    return (
                        torch.from_numpy(result["image"]),
                        torch.from_numpy(result["mask"]).to(torch.float32),
                    )
            else:
                return (
                    result["image"],
                    result["mask"].to(torch.float32),
                )
        else:
            return (
                torch.from_numpy(image).permute(2, 0, 1),
                torch.from_numpy(mask).to(torch.float32),
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self._read_image_(self.image_paths[index])
        mask = self._read_mask_(self.mask_paths[index])
        image_tensor, mask_tensor = self._process_image_mask_(image, mask)

        assert (image_tensor.shape[1], image_tensor.shape[2]) == (
            mask_tensor.shape[0],
            mask_tensor.shape[1],
        ), f"Image H, W ({(image_tensor.shape[1], image_tensor.shape[2])}) and mask H, W ({(mask_tensor.shape[0], mask_tensor.shape[1])}) mismatch."

        return image_tensor, mask_tensor

    def prepare_data(self) -> None:
        pass


if __name__ == "__main__":
    train_dataset = KvasirSEGDataset("./data/", mode="train")
    val_dataset = KvasirSEGDataset("./data/", mode="val")
    test_dataset = KvasirSEGDataset("./data/", mode="test")
    for i in range(len(train_dataset)):
        img, mask = train_dataset[i]
    # print(type(img), type(mask))
    print(
        train_dataset.__len__()
        + val_dataset.__len__()
        + test_dataset.__len__()
    )
