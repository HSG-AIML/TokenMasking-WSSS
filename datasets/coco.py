import json
from typing import Any, Callable, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader, RandomSampler
from torchvision import tv_tensors
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor
from torchvision.transforms.v2 import Compose
from torchvision.transforms.v2 import functional as F
from torchvision.tv_tensors import Mask
from torchvision.tv_tensors._dataset_wrapper import (
    list_of_dicts_to_dict_of_lists,
    parse_target_keys,
)

NUM_CLASSES = 80


class CocoLabelRemapper(torch.nn.Module):
    def __init__(self, ann_file: str):
        super().__init__()
        with open(ann_file, "r") as file:
            anns = json.load(file)

        self.categories = anns["categories"]
        self.category_id_map = {}
        for idx, cat in enumerate(self.categories):
            self.category_id_map[cat["id"]] = idx
            cat.update(id=idx, og_id=cat["id"])

    def forward(self, img, target):
        target["labels"] = torch.tensor(
            list(self.category_id_map[int(label)] for label in target["labels"])
        )
        return img, target


class InstanceSemanticMaskTransform(torch.nn.Module):
    def get_semantic_masks(self, masks: Mask, labels: Tensor) -> Tuple[Mask, Tensor]:
        """Convert instance masks to semantic masks (one per class)."""
        class_masks = {}
        for label, mask in zip(labels, masks):
            label = int(label)
            class_masks[label] = class_masks.get(label, []) + [mask]

        for label, masks in class_masks.items():
            class_masks[label] = sum(masks)

        labels = torch.tensor(list(class_masks.keys()))
        masks = Mask(torch.stack(list(class_masks.values())))
        return masks, labels

    def debinarize_mask(self, binary_masks, labels):
        c, w, h = binary_masks.shape

        result_mask = np.ones((w, h), dtype=int) * -1

        for i in range(c):
            result_mask[binary_masks[i] == 1] = labels[i]

        return torch.Tensor(result_mask)

    def forward(self, img, target):
        target["masks"], target["labels"] = self.get_semantic_masks(
            target["masks"], target["labels"]
        )
        target["masks"] = self.debinarize_mask(target["masks"], target["labels"])
        return img, target


class OutputReshaper(torch.nn.Module):
    def resize(self, img, mask):
        img = cv2.resize(img, (448, 448), interpolation=cv2.INTER_NEAREST)
        print(img.shape)
        mask = cv2.resize(mask, (448, 448), interpolation=cv2.INTER_NEAREST)
        print(mask.shape)

        return img, mask

    def one_hot_encode(self, label):
        one_hot = torch.zeros(NUM_CLASSES)

        one_hot[label] = 1.0

        return one_hot

    def forward(self, img, target):
        img = np.array(img)
        mask = np.array(target["masks"])

        img, mask = self.resize(img, mask)
        output_tensor = {
            "image_id": target["image_id"],
            "img": ToTensor()(img),
            "y": self.one_hot_encode(target["labels"]),
            "y_ohe": self.one_hot_encode(target["labels"]),
            "masks": mask,
        }

        return output_tensor


class CocoSegmentation(CocoDetection):
    def __init__(
        self,
        img_dir,
        ann_file,
        target_keys,
        transforms=None,
    ):
        super().__init__(img_dir, ann_file, transforms=transforms)

        with open(ann_file, "r") as file:
            anns = json.load(file)

        self.ids = []
        seen = set()
        seen_add = seen.add
        self.ids = [
            ann["image_id"]
            for ann in anns["annotations"]
            if not (ann["image_id"] in seen or seen_add(ann["image_id"]))
        ]

        self.target_keys = parse_target_keys(
            target_keys,
            available={
                # native
                "segmentation",
                "area",
                "iscrowd",
                "image_id",
                "bbox",
                "category_id",
                # added by the wrapper
                "boxes",
                "masks",
                "labels",
            },
            default={"image_id", "masks", "labels"},
        )

    @classmethod
    def from_split(
        cls,
        split: str = "val",
        target_keys=None,
        transforms=None,
    ):
        if split == "val":
            img_dir = "/ds2/computer_vision/MS-COCO/images/val2017"
            ann_file = "/ds2/computer_vision/MS-COCO/annotations/instances_val2017.json"
        elif split == "train":
            img_dir = "/ds2/computer_vision/MS-COCO/images/train2017"
            ann_file = (
                "/ds2/computer_vision/MS-COCO/annotations/instances_train2017.json"
            )
        else:
            raise ValueError("Invalid split, must be 'train' or 'val'.")

        dataset = cls(img_dir, ann_file, target_keys=target_keys, transforms=transforms)
        return dataset

    def segmentation_to_mask(self, segmentation, *, canvas_size):
        from pycocotools import mask

        if isinstance(segmentation, dict):
            # if counts is a string, it is already an encoded RLE mask
            if not isinstance(segmentation["counts"], str):
                segmentation = mask.frPyObjects(segmentation, *canvas_size)
        elif isinstance(segmentation, list):
            segmentation = mask.merge(mask.frPyObjects(segmentation, *canvas_size))
        else:
            raise ValueError(
                f"COCO segmentation expected to be a dict or a list, got {type(segmentation)}"
            )
        return torch.from_numpy(mask.decode(segmentation))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if not isinstance(index, int):
            raise ValueError(
                f"Index must be of type integer, got {type(index)} instead."
            )

        image_id = self.ids[index]
        image = self._load_image(image_id)
        target = self._load_target(image_id)

        if not target:
            return image, dict(image_id=image_id)

        canvas_size = tuple(F.get_size(image))

        batched_target = list_of_dicts_to_dict_of_lists(target)
        target = {}

        if "image_id" in self.target_keys:
            target["image_id"] = image_id

        if "boxes" in self.target_keys:
            target["boxes"] = F.convert_bounding_box_format(
                tv_tensors.BoundingBoxes(
                    batched_target["bbox"],
                    format=tv_tensors.BoundingBoxFormat.XYWH,
                    canvas_size=canvas_size,
                ),
                new_format=tv_tensors.BoundingBoxFormat.XYXY,
            )

        if "masks" in self.target_keys:
            target["masks"] = Mask(
                torch.stack(
                    [
                        self.segmentation_to_mask(segmentation, canvas_size=canvas_size)
                        for segmentation in batched_target["segmentation"]
                    ]
                ),
            )

        if "labels" in self.target_keys:
            target["labels"] = torch.tensor(batched_target["category_id"])

        for target_key in self.target_keys - {"image_id", "boxes", "masks", "labels"}:
            target[target_key] = batched_target[target_key]

        if self.transforms:
            output_tensor = self.transforms(image, target)

        return output_tensor


class COCODataModule(pl.LightningDataModule):
    def __init__(
        self,
    ):
        super(COCODataModule).__init__()

    def prepare_data(self):
        pass

    def setup(self, train_batch_size, eval_batch_size):
        label_remapper = CocoLabelRemapper(
            "/ds2/computer_vision/MS-COCO/annotations/instances_val2017.json"
        )
        transforms_val = Compose(
            [InstanceSemanticMaskTransform(), label_remapper, OutputReshaper()]
        )

        transforms_train = Compose(
            [InstanceSemanticMaskTransform(), label_remapper, OutputReshaper()]
        )

        self.trainset = CocoSegmentation.from_split(
            "train", transforms=transforms_train
        )

        self.valset = CocoSegmentation.from_split("val", transforms=transforms_val)

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
