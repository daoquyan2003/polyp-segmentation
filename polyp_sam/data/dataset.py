import os

import cv2
import numpy as np
import torch
from segment_anything.utils.transforms import ResizeLongestSide
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ColonDataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_size):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        gt = cv2.imread(self.mask_paths[index], cv2.IMREAD_GRAYSCALE) / 255
        gt = gt.astype("float32")

        bbox_arr = self.extract_bboxes(gt, 1)
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

    @staticmethod
    def extract_bboxes(mask, num_instances):
        boxes = np.zeros([num_instances, 4], dtype=np.int32)
        for i in range(num_instances):
            m = mask
            horizontal_indicies = np.where(np.any(m, axis=0))[0]
            vertical_indicies = np.where(np.any(m, axis=1))[0]
            if horizontal_indicies.shape[0]:
                x1, x2 = horizontal_indicies[[0, -1]]
                y1, y2 = vertical_indicies[[0, -1]]
                x2 += 1
                y2 += 1
            else:
                x1, x2, y1, y2 = 0, 0, 0, 0
            boxes[i] = np.array([y1, x1, y2, x2])
        return boxes.astype(np.int32)


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


def create_data_loaders(image_folder, mask_folder, batch_size, image_size):
    image_paths = []
    mask_paths = []

    for root, _, files in os.walk(image_folder):
        for name in files:
            if name.endswith(".png"):
                image_paths.append(os.path.join(root, name))

    for root, _, files in os.walk(mask_folder):
        for name in files:
            if name.endswith(".png"):
                mask_paths.append(os.path.join(root, name))

    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )

    train_dataset = ColonDataset(X_train, y_train, image_size)
    val_dataset = ColonDataset(X_test, y_test, image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=my_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=my_collate,
    )

    return train_loader, val_loader
