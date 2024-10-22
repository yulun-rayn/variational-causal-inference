import os
import copy
from PIL import Image
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import torch
import torchvision.transforms.functional as F
from torchvision import datasets, transforms
from torchvision.datasets.utils import verify_str_arg

from .base_dataset import BaseDataset

from ..utils.data_utils import AttrEncoder


class ImageDataset(BaseDataset):
    def __init__(self, root: str, img_folder: str = "img", load_fn: Callable = Image.open,
                 attr_file: str = "attr.tsv", split_file: str = "split.tsv",
                 label_names: List[str] = ["age", "sex"],
                 image_size: Tuple[int] = (192, 160), pad_size: int = 8, split: str = "train"):
        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all"))]
        splits = pd.read_csv(os.path.join(root, split_file), index_col=0, sep="\t")
        attr = pd.read_csv(os.path.join(root, attr_file), index_col=0, sep="\t")

        images = splits.index
        labels = attr[label_names].values

        transform = self.get_transform(image_size, pad_size)
        target_transform = self.get_target_transform(labels)

        if split_ is not None:
            mask = (splits.values == split_).squeeze()
            images = images[mask]
            labels = labels[mask]

        images = [os.path.join(root, img_folder, image) for image in images]
        super().__init__(images, labels, transform=transform, target_transform=target_transform, load_fn=load_fn)

    def get_transform(self, image_size, pad_size):
        return transforms.Compose([
            transforms.Resize((image_size[0] - pad_size, image_size[1] - pad_size)),
            transforms.RandomCrop(image_size, padding=pad_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])

    def get_target_transform(self, labels):
        return AttrEncoder(labels)
