import os
from PIL import Image
from typing import Optional

import numpy as np
import pandas as pd

import torch
from torchvision import datasets, transforms

from .base_dataset import BaseDataset

from ..utils.general_utils import load_idx
from ..utils.data_utils import AttrEncoder


class MorphoMNISTDataset:
    def __init__(self, root, label_names=["thickness", "intensity"], image_size=(32, 32)):
        self.root = root
        self.label_names = label_names
        self.image_size = image_size

        self.images, self.attr = {}, {}
        self.images["train"], self.attr["train"] = self.get_data_df("train")
        self.images["test"], self.attr["test"] = self.get_data_df("test")
        full_attr = pd.concat((self.attr["train"], self.attr["test"]))

        self.target_transform = self.get_target_transform(full_attr.values)

    def get_split(self, split):
        return BaseDataset(
            self.images[split], self.attr[split].values,
            transform=self.get_transform(self.image_size, split),
            target_transform=self.target_transform
        )

    def get_data_df(self, split):
        if split == "train":
            path_to_images = os.path.join(self.root, "train-images-idx3-ubyte.gz")
            path_to_labels = os.path.join(self.root, "train-labels-idx1-ubyte.gz")
            path_to_attr = os.path.join(self.root, "train-morpho.csv")
        else:
            path_to_images = os.path.join(self.root, "t10k-images-idx3-ubyte.gz")
            path_to_labels = os.path.join(self.root, "t10k-labels-idx1-ubyte.gz")
            path_to_attr = os.path.join(self.root, "t10k-morpho.csv")

        images = load_idx(path_to_images)

        attrs = pd.read_csv(path_to_attr)
        attrs['label'] = load_idx(path_to_labels).astype(str)
        attrs.drop(list(attrs.filter(regex='Unnamed')), axis=1, inplace=True)
        attrs = attrs[self.label_names]

        return images, attrs

    def get_transform(self, image_size, split):
        if split == "train":
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop(image_size, padding=4)
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Pad(padding=2)
            ])

    def get_target_transform(self, labels):
        return AttrEncoder(labels)
