import os
from PIL import Image
from typing import Optional
from functools import partial

import numpy as np
import pandas as pd

import torch
from torchvision import datasets, transforms
from torchvision.datasets.utils import verify_str_arg

from .base_dataset import BaseDataset

from ..utils.data_utils import AttrEncoder


class CelebADataset(BaseDataset):
    base_folder = "celeba"

    def __init__(self, root, label_idx=[15, 31], image_size=(64, 64), split="train"):
        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all"))]

        fn = partial(os.path.join, root, self.base_folder)
        splits = pd.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        attr = pd.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)

        mask = slice(None) if split_ is None else (splits[1] == split_)

        # image
        images = splits.index

        transform = self.get_transform(image_size, split)

        images = images[mask]
        images = [fn("img_align_celeba", image) for image in images]

        # label
        labels = attr.values[:, label_idx]
        labels = labels > 0

        target_transform = self.get_target_transform(labels)

        labels = labels[mask]

        super().__init__(images, labels, transform=transform, target_transform=target_transform, load_fn=Image.open)

    def get_transform(self, image_size, split):
        if split == "train":
            return transforms.Compose([
                transforms.CenterCrop(128),
                transforms.RandomCrop((120, 120)),
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor()
            ])
        else:
            return transforms.Compose([
                transforms.CenterCrop(120),
                transforms.Resize(image_size),
                transforms.ToTensor()
            ])

    def get_target_transform(self, labels):
        return AttrEncoder(labels)
