import os
import copy
from collections.abc import Iterable
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import torch
import torchvision.transforms.functional as F


class BaseDataset:
    def __init__(self, values: Iterable[str, np.ndarray, torch.Tensor], labels: Union[List, np.ndarray, torch.Tensor],
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,
                 load_fn: Optional[Callable] = None) -> None:
        self.values = values
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.load_fn = load_fn

        sample_value = values[0]
        if load_fn:
            sample_value = load_fn(sample_value)
        if transform:
            sample_value = transform(sample_value)
        else:
            sample_value = torch.tensor(sample_value)

        sample_label = labels[0]
        if target_transform:
            sample_label = target_transform(sample_label)
        else:
            sample_label = torch.as_tensor(sample_label)

        self.num_outcomes = self.get_num_outcomes(sample_value)
        self.num_treatments = self.get_num_treatments(sample_label)
        self.num_covariates = self.get_num_covariates()

        self.nb_dims = sample_value.ndim

    def get_num_outcomes(self, value):
        return tuple(value.size())

    def get_num_treatments(self, label):
        return len(label)

    def get_num_covariates(self):
        return [1] # dummy covariate

    def __getitem__(self, index: int) -> Any:
        value = self.values[index]

        if self.load_fn:
            value = self.load_fn(value)
        if self.transform:
            value = self.transform(value)
        else:
            value = torch.tensor(value)

        label = self.labels[index]

        cf_index = np.random.choice(len(self.labels))
        cf_field = np.random.choice(len(label))
        cf_label = copy.deepcopy(label)
        cf_label[cf_field] = self.labels[cf_index][cf_field]

        if self.target_transform:
            label = self.target_transform(label)
            cf_label = self.target_transform(cf_label)
        else:
            label = torch.as_tensor(label)
            cf_label = torch.as_tensor(cf_label)

        return ( # Y, T, X, T', Ys under X and T'
            value, label, [torch.tensor([0.])], cf_label, None
        )

    def __len__(self) -> int:
        return len(self.labels)
