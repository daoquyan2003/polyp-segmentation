import cv2
import numpy as np
import torch
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import Dataset
from torchvision import transforms

from .utils.utils import extract_bboxes


class ColonDataset(Dataset):
    def __init__(self, image_path, mask_path, image_size):
        self.image_path = image_path
        self.mask_path = mask_path
        self.image_size = image_size

        # TODO: use ResizeLongestSide and pad to square
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        image = cv2.imread(self.image_path[index])
        gt = cv2.imread(self.mask_path[index])
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gt)
        mask[gt > 0] = 1
        gt = mask
        gt = gt.astype("float32")

        bbox_arr = extract_bboxes(gt, 1)

        gt_resized = cv2.resize(gt, (1024, 1024), cv2.INTER_NEAREST)
        gt_resized = torch.as_tensor(gt_resized > 0).long()

        gt = torch.from_numpy(gt)
        gt_binary_mask = torch.as_tensor(gt > 0).long()

        transform = ResizeLongestSide(self.image_size)
        input_image = transform.apply_image(image)
        input_image = cv2.resize(input_image, (1024, 1024), cv2.INTER_CUBIC)
        input_image = self.to_tensor(input_image)

        original_image_size = image.shape[:2]
        input_size = tuple(input_image.shape[-2:])

        return (
            input_image,
            np.array(bbox_arr),
            gt_binary_mask,
            gt_resized,
            original_image_size,
            input_size,
        )


def my_collate(batch):

    images, bboxes, masks, gt_resized, original_image_size, input_size = zip(
        *batch
    )
    images = torch.stack(images, dim=0)
    gt_resized = torch.stack(gt_resized, dim=0)

    masks = [m for m in masks]
    bboxes = [m for m in bboxes]
    original_image_size = [m for m in original_image_size]
    input_size = [m for m in input_size]

    return images, bboxes, masks, gt_resized, original_image_size, input_size
