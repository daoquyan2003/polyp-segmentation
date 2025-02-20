from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import random_split

from src.datamodules.components.neopolyp_dataset import NeoPolypDataset
from src.datamodules.components.transforms import TransformsWrapper
from src.datamodules.datamodules import SingleDataModule


class NeoPolypDataModule(SingleDataModule):
    def __init__(
        self, datasets: DictConfig, loaders: DictConfig, transforms: DictConfig
    ) -> None:
        super.__init__(
            datasets=datasets, loaders=loaders, transforms=transforms
        )

    @property
    def num_classes(self) -> int:
        return 3

    def prepare_data(self) -> None:
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        NeoPolypDataset(self.cfg_datasets.get("data_dir")).prepare_data()

    def setup(self, stage: str | None = None) -> None:
        # load and split datasets only if not loaded already
        if not self.train_set and not self.valid_set and not self.test_set:
            transforms_train = TransformsWrapper(self.transforms.get("train"))
            transforms_test = TransformsWrapper(
                self.transforms.get("valid_test_predict")
            )

            train_dataset = NeoPolypDataset(
                self.cfg_datasets.get("data_dir"),
                transforms=transforms_train,
            )

            test_dataset = NeoPolypDataset(
                self.cfg_datasets.get("data_dir"),
                transforms=transforms_test,
            )

            seed = self.cfg_datasets.get("seed")
            self.train_set, _, _ = random_split(
                dataset=train_dataset,
                lengths=self.cfg_datasets.get("train_val_test_split"),
                generator=torch.Generator().manual_seed(seed),
            )

            _, self.valid_set, self.test_set = random_split(
                dataset=test_dataset,
                lengths=self.cfg_datasets.get("train_val_test_split"),
                generator=torch.Generator().manual_seed(seed),
            )

        # load predict dataset only if test set existed already
        if (stage == "predict") and self.test_set:
            self.predict_set = {"PredictDataset": self.test_set}

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
