from typing import Any

import albumentations
import hydra
import numpy as np
from omegaconf import DictConfig
from PIL import Image


class TransformsWrapper:
    def __init__(self, transforms_cfg: DictConfig) -> None:
        """TransformsWrapper module.

        Args:
            transforms_cfg (DictConfig): Transforms config.
        """

        augmentations = []
        if not transforms_cfg.get("order"):
            raise RuntimeError(
                "TransformsWrapper requires param <order>, i.e."
                "order of augmentations as List[augmentation name]"
            )
        for augmentation_name in transforms_cfg.get("order"):
            augmentation = hydra.utils.instantiate(
                transforms_cfg.get(augmentation_name), _convert_="object"
            )
            augmentations.append(augmentation)
        self.augmentations = albumentations.Compose(
            augmentations,
            bbox_params=albumentations.BboxParams(
                format="pascal_voc", label_fields=["dummy_labels"]
            ),
        )

    def __call__(
        self, image: Any, mask: Any | None = None, **kwargs: Any
    ) -> Any:
        """Apply TransformsWrapper module.

        Args:
            image (Any): Input image.
            kwargs (Any): Additional arguments.

        Returns:
            Any: Transformation results.
        """

        if isinstance(image, Image.Image):
            image = np.asarray(image)
        if mask is not None and isinstance(mask, Image.Image):
            mask = np.asarray(mask)

        if mask is not None:
            return self.augmentations(image=image, mask=mask, **kwargs)
        else:
            image = self.augmentations(image=image, **kwargs)["image"]
            return {"image": image, "mask": None}
