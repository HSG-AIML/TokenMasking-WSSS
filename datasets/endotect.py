from __future__ import annotations

import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler

NUM_CLASS = 16

CLASSES_MAP = {
    "bbps-2-3": 0,
    "polyps": 1,
    "cecum": 2,
    "dyed-lifted-polyps": 3,
    "dyed-resection-margins": 4,
    "bbps-0-1": 5,
    "ulcerative-colitis-grade-2": 6,
    "retroflex-rectum": 7,
    "ulcerative-colitis-grade-1": 8,
    "ulcerative-colitis-grade-3": 9,
    "impacted-stool": 10,
    "ulcerative-colitis-grade-0-1": 11,
    "ulcerative-colitis-grade-2-3": 12,
    "ulcerative-colitis-grade-1-2": 13,
    "ileum": 14,
    "hemorrhoids": 15,
}


class Endotect:
    """Endotect Dataset."""

    def __init__(
        self,
        split,
        datadir="/ds2/computer_vision/endotect",
        transform=False,
        mult=1,
        **kwargs,
    ):
        super(Endotect, self).__init__()
        self.images, self.masks = _get_endotect_pairs(datadir, split)

        assert len(self.images) == len(self.masks)

        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + datadir + "\n")

        print("Found {} images in the folder {}".format(len(self.images), datadir))

        self.transforms = transform

        if self.transforms:
            augmentation = A.Compose(
                [
                    A.Resize(
                        height=512, width=640, p=1, interpolation=cv2.INTER_NEAREST
                    ),
                    A.augmentations.transforms.CLAHE(p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    A.Affine(scale=1.5, translate_px=5, rotate=20, p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    ToTensorV2(),
                ]
            )
            self.augmentation = augmentation

        else:
            augmentation = A.Compose(
                [
                    A.Resize(
                        height=512, width=640, p=1, interpolation=cv2.INTER_NEAREST
                    ),
                    ToTensorV2(),
                ]
            )
            self.augmentation = augmentation

        # turn lists into arrays
        self.images = np.array(self.images)
        self.masks = np.array(self.masks)

        # increase data set size by factor `mult`
        if mult > 1:
            self.images = np.array([*self.images] * mult)
            self.masks = np.array([*self.masks] * mult)

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        if self.masks[index]:
            mask = cv2.imread(self.masks[index])
        else:
            mask = np.zeros((512, 640, 3))

        label = CLASSES_MAP[self.images[index].split("/")[7]]
        label_ohe = np.zeros((NUM_CLASS, 1))
        label_ohe[label] = 1

        augmented_data = self.augmentation(image=img, mask=mask)
        img, mask = augmented_data["image"], augmented_data["mask"]

        # min max scaler
        img = self.minmax(img)

        output_tensor = {
            "maskfile": self.masks[index],
            "img": img,  # torch.unsqueeze(img[0], 0), #img
            "imgfile": self.images[index],
            "mask": mask.to(torch.uint8),
            "y": int(label),
            "y_ohe": torch.from_numpy(label_ohe),
        }

        return output_tensor

    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype("int32") - 1)

    def minmax(self, img):
        return (img - img.min()) / (img.max() - img.min())

    def __len__(self):
        return len(self.images)


def _get_endotect_pairs(data_folder, split):
    img_paths = []
    mask_paths = []

    img_folder = os.path.join(data_folder, "labeled-images")
    mask_folder = os.path.join(data_folder, "segmented-images")

    for root, _, files in os.walk(img_folder):
        for filename in files:
            if (
                filename.endswith(".jpg")
                and "lower-gi-tract" in root
                and filename[:-4] in split
            ):
                imgpath = os.path.join(root, filename)
                maskpath = os.path.join(mask_folder, "masks", filename)
                img_paths.append(imgpath)
                if os.path.isfile(maskpath):
                    mask_paths.append(maskpath)
                else:
                    mask_paths.append("")

    return img_paths, mask_paths


class EndotectDataModule(pl.LightningDataModule):
    def __init__(
        self,
    ):
        super(EndotectDataModule).__init__()

    def prepare_data(self):
        pass

    def setup(self, train_batch_size, eval_batch_size):
        df = pd.read_csv(
            os.path.join(
                "/ds2/computer_vision/endotect/", "labeled-images/image-labels.csv"
            )
        )
        names = df[df["Organ"] == "Lower GI"]["Video file"].tolist()
        X_train, X_val = train_test_split(names, test_size=0.3, random_state=42)

        self.trainset = Endotect(
            split=X_train,
            datadir="/ds2/computer_vision/endotect/",
            transform=True,
            seed=42,
        )

        self.valset = Endotect(
            split=X_val,
            datadir="/ds2/computer_vision/endotect/",
            transform=False,
            seed=42,
        )

        self.train_bs = train_batch_size
        self.eval_bs = eval_batch_size

    def train_dataloader(self):
        random_sampler = RandomSampler(self.trainset)

        return DataLoader(
            self.trainset,
            batch_size=self.train_bs,
            num_workers=4,
            pin_memory=True,
            sampler=random_sampler,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valset, batch_size=self.eval_bs, num_workers=4, pin_memory=True
        )
