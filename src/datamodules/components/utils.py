from operator import itemgetter
from typing import Iterator, Optional

import numpy as np
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler


class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.

    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.

        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """Wrapper over `Sampler` for distributed training. Allows you to use any
    sampler in distributed mode. It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each process can
    pass a DistributedSamplerWrapper instance as a DataLoader sampler, and load
    a subset of subsampled data of the original dataset that is exclusive to
    it.

    .. note::     Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super().__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.

        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        sub_sampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(sub_sampler_indexes))


def extract_bboxes_xyxy(mask, num_instances):
    """Compute bounding boxes from masks.

    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (x1, y1, x2, y2)].
    """

    boxes = np.zeros([num_instances, 4], dtype=np.int32)
    processed_mask = np.zeros_like(mask)
    processed_mask[mask > 0] = 1
    for i in range(num_instances):
        m = processed_mask

        # Find bounding box coordinates
        horizontal_indices = np.where(np.any(m, axis=0))[0]  # Get x-range
        vertical_indices = np.where(np.any(m, axis=1))[0]  # Get y-range

        if horizontal_indices.shape[0]:
            x1, x2 = horizontal_indices[[0, -1]]
            y1, y2 = vertical_indices[[0, -1]]

            # Ensure x2, y2 are included in the bbox
            x2 += 1
            y2 += 1

        else:
            # No mask for this instance
            x1, y1, x2, y2 = 0, 0, 0, 0

        boxes[i] = np.array([x1, y1, x2, y2])

    return boxes.astype(np.int32)
