"""Pascal VOC Semantic Segmentation Dataset."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import albumentations as A
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler

VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "potted plant",
    "sheep",
    "sofa",
    "train",
    "tv/monitor",
    # "background",
]


VOC_COLORMAP = [
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
    # [0, 0, 0],
]


VOC_COLORS = (
    "#800000",
    "#008000",
    "#808000",
    "#000080",
    "#800080",
    "#008080",
    "#808080",
    "#400000",
    "#C00000",
    "#408000",
    "#C08000",
    "#400080",
    "#C00080",
    "#408080",
    "#C08080",
    "#004000",
    "#804000",
    "#00C000",
    "#80C000",
    "#004080",
    # "#000000",
)


VOC_CLASSES_DICT = {
    "aeroplane": 0,
    "bicycle": 1,
    "bird": 2,
    "boat": 3,
    "bottle": 4,
    "bus": 5,
    "car": 6,
    "cat": 7,
    "chair": 8,
    "cow": 9,
    "diningtable": 10,
    "dog": 11,
    "horse": 12,
    "motorbike": 13,
    "person": 14,
    "pottedplant": 15,
    "sheep": 16,
    "sofa": 17,
    "train": 18,
    "tvmonitor": 19,
}


class VOCSegmentation:
    """VOC Semantic Segmentation Dataset."""

    NUM_CLASS = 20

    CLASSES = {
        "person": 0,
        "bird": 1,
        "cat": 2,
        "cow": 3,
        "dog": 4,
        "horse": 5,
        "sheep": 6,
        "aeroplane": 7,
        "bicycle": 8,
        "boat": 9,
        "bus": 10,
        "car": 11,
        "motorbike": 12,
        "train": 13,
        "bottle": 14,
        "chair": 15,
        "diningtable": 16,
        "pottedplant": 17,
        "sofa": 18,
        "tvmonitor": 19,
    }

    def __init__(
        self,
        split=[],
        root="/local/VOC2012/",
        transform=True,
        mult=1,
        imgsize=256,
        **kwargs,
    ):
        super(VOCSegmentation, self).__init__()
        self.images, self.masks, self.labels = _get_voc_pairs(root, split)
        assert len(self.images) == len(self.masks) == len(self.labels)
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")

        print("Found {} images in the folder {}".format(len(self.images), root))

        self.transforms = transform
        self.imgsize = imgsize

        if self.transforms:
            print("Augmentation")
            self.augmentation = A.Compose(
                [
                    A.Normalize(),
                    A.Resize(
                        height=self.imgsize,
                        width=self.imgsize,
                        p=1,
                        interpolation=cv2.INTER_NEAREST,
                    ),
                    A.Affine(scale=1, translate_px=5, rotate=20, p=0.3),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    ToTensorV2(),
                ]
            )

        else:
            print("No Augmentation")
            self.augmentation = A.Compose(
                [
                    A.Normalize(),
                    A.Resize(
                        height=self.imgsize,
                        width=self.imgsize,
                        p=1,
                        interpolation=cv2.INTER_NEAREST,
                    ),
                    ToTensorV2(),
                ]
            )

    @staticmethod
    def _convert_to_segmentation_mask(mask):
        # This function converts a mask from the Pascal VOC format to the format required by AutoAlbument.
        #
        # Pascal VOC uses an RGB image to encode the segmentation mask for that image. RGB values of a pixel
        # encode the pixel's class.
        #
        # AutoAlbument requires a segmentation mask to be a NumPy array with the shape [height, width, num_classes].
        # Each channel in this mask should encode values for a single class. Pixel in a mask channel should have
        # a value of 1.0 if the pixel of the image belongs to this class and 0.0 otherwise.
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros(
            (height, width, len(VOC_COLORMAP)), dtype=np.float32
        )
        for label_index, label in enumerate(VOC_COLORMAP):
            segmentation_mask[:, :, label_index] = np.all(
                mask == label, axis=-1
            ).astype(float)
        return segmentation_mask

    def __getitem__(self, index):
        img = cv2.imread(self.images[index])
        mask = cv2.imread(self.masks[index])
        labelxml = ET.parse(self.labels[index])

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self._convert_to_segmentation_mask(mask)

        root = labelxml.getroot()
        labelarray = []
        for object in root.findall("object"):
            labelarray.append(VOC_CLASSES_DICT[object[0].text])

        label = np.zeros(self.NUM_CLASS)
        label[np.array(labelarray)] = 1

        augmented = self.augmentation(image=img, mask=mask)

        img = augmented["image"].float()
        mask = augmented["mask"]

        output_tensor = {
            "img": img,
            "imgfile": os.path.basename(self.images[index]),
            "mask": mask,
            "y": torch.from_numpy(label.copy()),
            "y_ohe": torch.from_numpy(label.copy()),
        }

        return output_tensor

    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype("int32") - 1)

    def __len__(self):
        return len(self.images)

    @property
    def classes(self):
        """Category names."""
        return (
            "person",
            "bird",
            "cat",
            "cow",
            "dog",
            "horse",
            "sheep",
            "aeroplane",
            "bicycle",
            "boat",
            "bus",
            "car",
            "motorbike",
            "train",
            "bottle",
            "chair",
            "diningtable",
            "pottedplant",
            "sofa",
            "tvmonitor",
        )

    @property
    def colors(self):
        return (
            "#000000",
            "#408080",
            "#808000",
            "#808080",
            "#C00000",
            "#C08000",
            "#400080",
            "#004000",
            "#800000",
            "#008000",
            "#000080",
            "#008080",
            "#004080",
            "#C00080",
            "#00C000",
            "#800080",
            "#400000",
            "#408000",
            "#C08080",
            "#804000",
            "#80C000",
        )


def _get_voc_pairs(folder, files):
    img_paths = []
    mask_paths = []
    label_paths = []

    img_folder = os.path.join(folder, "JPEGImages")
    mask_folder = os.path.join(folder, "SegmentationClass")
    label_folder = os.path.join(folder, "Annotations")

    for filename in files:
        basename = filename.strip("\n")
        imgpath = os.path.join(img_folder, basename + ".jpg")
        maskname = basename + ".png"
        maskpath = os.path.join(mask_folder, maskname)
        labelname = basename + ".xml"
        labelpath = os.path.join(label_folder, labelname)
        if os.path.isfile(maskpath):
            img_paths.append(imgpath)
            mask_paths.append(maskpath)
            label_paths.append(labelpath)
        else:
            print("cannot find the mask:", maskpath)

    return img_paths, mask_paths, label_paths


class VOCDataModule(pl.LightningDataModule):
    def __init__(
        self,
    ):
        super(VOCDataModule).__init__()

    def prepare_data(self):
        pass

    def setup(self, train_batch_size, eval_batch_size):
        datadir = "/local/VOC2012/"

        with open(os.path.join(datadir, "ImageSets/Segmentation/trainval.txt")) as f:
            names = f.readlines()

        X_train, X_val = train_test_split(names, test_size=0.3, random_state=42)

        self.trainset = VOCSegmentation(
            split=X_train,
            root=datadir,
            imgsize=512,
            mult=1,
            transform=True,
        )

        self.valset = VOCSegmentation(
            split=X_val,
            root=datadir,
            imgsize=512,
            transform=None,
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
