import glob
import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

with open("kaggle.json") as f:
    data = json.load(f)

os.environ["KAGGLE_USERNAME"] = data["username"]
os.environ["KAGGLE_KEY"] = data["key"]

from kaggle.api.kaggle_api_extended import KaggleApi


class NeoPolypDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        transforms: Callable | None = None,
        read_mode: str = "pillow",
    ) -> None:
        self.data_dir = data_dir
        self.transforms = transforms
        self.read_mode = read_mode

        image_dir = os.path.join(self.data_dir, "train", "train")
        mask_dir = os.path.join(self.data_dir, "train_gt", "train_gt")

        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpeg")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.jpeg")))

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
        processed_mask[(mask >= 50) & (mask <= 100)] = 1
        processed_mask[mask > 100] = 2

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
            result = self.transforms(image=image, mask=mask)
            if isinstance(result["image"], np.ndarray):
                if result["image"].ndim == 3 and result["image"].shape[2] == 3:
                    return torch.from_numpy(result["image"]).permute(
                        2, 0, 1
                    ), torch.from_numpy(result["mask"])
                else:
                    return torch.from_numpy(result["image"]), torch.from_numpy(
                        result["mask"]
                    )
            else:
                return result["image"], result["mask"]
        else:
            return torch.from_numpy(image).permute(2, 0, 1), torch.from_numpy(
                mask
            )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self._read_image_(self.image_paths[index])
        mask = self._read_mask_(self.mask_paths[index])
        image_tensor, mask_tensor = self._process_image_mask_(image, mask)
        return image_tensor, mask_tensor

    def prepare_data(self) -> None:
        """Prepare the dataset by downloading and extracting files if not
        already present.

        This function checks if the training data directory exists. If not, it uses the Kaggle API
        to download the dataset from the specified competition, extracts the contents,
        and performs cleanup by removing unnecessary files.

        Raises:
            KaggleApiException: If there is an issue with Kaggle API authentication or downloading.
        """

        data_path = os.path.join(self.data_dir, "train")
        if os.path.exists(data_path):
            print("Data is already downloaded.")
            return

        api = KaggleApi()
        api.authenticate()

        competition = "bkai-igh-neopolyp"

        os.makedirs(self.data_dir, exist_ok=True)

        print("Downloading data...")
        api.competition_download_files(
            competition, path=self.data_dir, quiet=False
        )

        downloaded_file = os.path.join(self.data_dir, "bkai-igh-neopolyp.zip")

        print("Unzipping data...")
        with zipfile.ZipFile(downloaded_file, "r") as zip_ref:
            zip_ref.extractall(self.data_dir)

        print("Removing unnecessary files...")
        os.remove(downloaded_file)
        os.remove(os.path.join(self.data_dir, "sample_submission.csv"))

        print("Done.")


if __name__ == "__main__":
    dataset = NeoPolypDataset("./data/")
    img, mask = dataset[0]
    print(img.shape, mask.shape)
    print(type(img), type(mask))
