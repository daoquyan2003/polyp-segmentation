from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset

from src.datamodules.components.cvc_clinicdb_dataset import CVCClinicDBDataset
from src.datamodules.components.cvc_colondb_dataset import CVCColonDBDataset
from src.datamodules.components.etis_larib_polypdb_dataset import (
    ETISLaribPolypDBDataset,
)
from src.datamodules.components.kvasir_seg_dataset import KvasirSEGDataset
from src.datamodules.components.neopolyp_dataset_v1 import NeoPolypDataset
from src.datamodules.components.polypgen_dataset import PolypGenDataset
from src.datamodules.components.transforms import TransformsWrapper
from src.datamodules.datamodules import SingleDataModule


class PolypDataModule(SingleDataModule):
    def __init__(
        self, datasets: DictConfig, loaders: DictConfig, transforms: DictConfig
    ) -> None:
        super().__init__(
            datasets=datasets, loaders=loaders, transforms=transforms
        )

    @property
    def num_classes(self) -> int:
        return 2

    def prepare_data(self) -> None:
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: str | None = None) -> None:
        # load and split datasets only if not loaded already
        if not self.train_set and not self.valid_set and not self.test_set:
            transforms_train = TransformsWrapper(self.transforms.get("train"))
            transforms_test = TransformsWrapper(
                self.transforms.get("valid_test_predict")
            )

            neopolyp_train_dataset = NeoPolypDataset(
                self.cfg_datasets.get("data_dir"),
                transforms=transforms_train,
                train_val_test_ratio=self.cfg_datasets.get(
                    "train_val_test_split"
                ),
                mode="train",
            )

            neopolyp_val_dataset = NeoPolypDataset(
                self.cfg_datasets.get("data_dir"),
                transforms=transforms_test,
                train_val_test_ratio=self.cfg_datasets.get(
                    "train_val_test_split"
                ),
                mode="val",
            )

            neopolyp_test_dataset = NeoPolypDataset(
                self.cfg_datasets.get("data_dir"),
                transforms=transforms_test,
                train_val_test_ratio=self.cfg_datasets.get(
                    "train_val_test_split"
                ),
                mode="test",
            )

            kvasir_seg_train_dataset = KvasirSEGDataset(
                self.cfg_datasets.get("data_dir"),
                transforms=transforms_train,
                train_val_test_ratio=self.cfg_datasets.get(
                    "train_val_test_split"
                ),
                mode="train",
            )

            kvasir_seg_val_dataset = KvasirSEGDataset(
                self.cfg_datasets.get("data_dir"),
                transforms=transforms_test,
                train_val_test_ratio=self.cfg_datasets.get(
                    "train_val_test_split"
                ),
                mode="val",
            )

            kvasir_seg_test_dataset = KvasirSEGDataset(
                self.cfg_datasets.get("data_dir"),
                transforms=transforms_test,
                train_val_test_ratio=self.cfg_datasets.get(
                    "train_val_test_split"
                ),
                mode="test",
            )

            polypgen_train_dataset = PolypGenDataset(
                self.cfg_datasets.get("data_dir"),
                transforms=transforms_train,
                train_val_test_ratio=self.cfg_datasets.get(
                    "train_val_test_split"
                ),
                mode="train",
            )

            polypgen_val_dataset = PolypGenDataset(
                self.cfg_datasets.get("data_dir"),
                transforms=transforms_test,
                train_val_test_ratio=self.cfg_datasets.get(
                    "train_val_test_split"
                ),
                mode="val",
            )

            polypgen_test_dataset = PolypGenDataset(
                self.cfg_datasets.get("data_dir"),
                transforms=transforms_test,
                train_val_test_ratio=self.cfg_datasets.get(
                    "train_val_test_split"
                ),
                mode="test",
            )

            cvc_clinicdb_train_dataset = CVCClinicDBDataset(
                self.cfg_datasets.get("data_dir"),
                transforms=transforms_train,
                train_val_test_ratio=self.cfg_datasets.get(
                    "train_val_test_split"
                ),
                mode="train",
            )

            cvc_clinicdb_val_dataset = CVCClinicDBDataset(
                self.cfg_datasets.get("data_dir"),
                transforms=transforms_test,
                train_val_test_ratio=self.cfg_datasets.get(
                    "train_val_test_split"
                ),
                mode="val",
            )

            cvc_clinicdb_test_dataset = CVCClinicDBDataset(
                self.cfg_datasets.get("data_dir"),
                transforms=transforms_test,
                train_val_test_ratio=self.cfg_datasets.get(
                    "train_val_test_split"
                ),
                mode="test",
            )

            cvc_colondb_dataset = CVCColonDBDataset(
                self.cfg_datasets.get("data_dir"),
                transforms=transforms_test,
            )

            etis_larib_dataset = ETISLaribPolypDBDataset(
                self.cfg_datasets.get("data_dir"),
                transforms=transforms_test,
            )

            seed = self.cfg_datasets.get("seed")
            self.train_set = ConcatDataset(
                [
                    neopolyp_train_dataset,
                    kvasir_seg_train_dataset,
                    polypgen_train_dataset,
                    cvc_clinicdb_train_dataset,
                ]
            )
            self.valid_set = ConcatDataset(
                [
                    neopolyp_val_dataset,
                    kvasir_seg_val_dataset,
                    polypgen_val_dataset,
                    cvc_clinicdb_val_dataset,
                ]
            )
            self.test_set = ConcatDataset(
                [
                    neopolyp_test_dataset,
                    kvasir_seg_test_dataset,
                    polypgen_test_dataset,
                    cvc_clinicdb_test_dataset,
                ]
            )
            print("Number of train samples:", len(self.train_set))
            print("Number of val samples:", len(self.valid_set))
            print("Number of test samples:", len(self.test_set))

        # load predict dataset only if test set existed already
        if (stage == "predict") and self.test_set:
            self.predict_set = {"PredictDataset": self.test_set}

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
