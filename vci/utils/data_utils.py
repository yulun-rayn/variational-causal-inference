
import re
import math
import collections

import numpy as np
import pandas as pd

from sklearn import preprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F

np_str_obj_array_pattern = re.compile(r'[SaUO]')

data_collate_err_msg_format = (
    "data_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")

def data_collate(batch, nb_dims=1):
    r"""
        Function that takes in a batch of data and puts the elements within the batch
        into a tensor with an additional outer dimension - batch size. The exact output type can be
        a :class:`torch.Tensor`, a `Sequence` of :class:`torch.Tensor`, a
        Collection of :class:`torch.Tensor`, or left unchanged, depending on the input type.
        This is used as the default function for collation when
        `batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.

        Here is the general input type (based on the type of the element within the batch) to output type mapping:
        * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
        * NumPy Arrays -> :class:`torch.Tensor`
        * `float` -> :class:`torch.Tensor`
        * `int` -> :class:`torch.Tensor`
        * `str` -> `str` (unchanged)
        * `bytes` -> `bytes` (unchanged)
        * `Mapping[K, V_i]` -> `Mapping[K, default_collate([V_1, V_2, ...])]`
        * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[default_collate([V1_1, V1_2, ...]), default_collate([V2_1, V2_2, ...]), ...]`
        * `Sequence[V1_i, V2_i, ...]` -> `Sequence[default_collate([V1_1, V1_2, ...]), default_collate([V2_1, V2_2, ...]), ...]`

        Args:
            batch: a single batch to be collated

        Examples:
            >>> # Example with a batch of `int`s:
            >>> default_collate([0, 1, 2, 3])
            tensor([0, 1, 2, 3])
            >>> # Example with a batch of `str`s:
            >>> default_collate(['a', 'b', 'c'])
            ['a', 'b', 'c']
            >>> # Example with `Map` inside the batch:
            >>> default_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
            {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
            >>> # Example with `NamedTuple` inside the batch:
            >>> Point = namedtuple('Point', ['x', 'y'])
            >>> default_collate([Point(0, 0), Point(1, 1)])
            Point(x=tensor([0, 1]), y=tensor([0, 1]))
            >>> # Example with `Tuple` inside the batch:
            >>> default_collate([(0, 1), (2, 3)])
            [tensor([0, 2]), tensor([1, 3])]
            >>> # Example with `List` inside the batch:
            >>> default_collate([[0, 1], [2, 3]])
            [tensor([0, 2]), tensor([1, 3])]
    """
    elem = batch[0]
    elem_type = type(elem)
    if elem is None:
        return list(batch)
    if isinstance(elem, torch.Tensor):
        if elem.dim() > nb_dims:
            return list(batch)

        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(data_collate_err_msg_format.format(elem.dtype))

            return data_collate([torch.as_tensor(b) for b in batch], nb_dims=nb_dims)
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: data_collate([d[key] for d in batch], nb_dims=nb_dims) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: data_collate([d[key] for d in batch], nb_dims=nb_dims) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(data_collate(samples, nb_dims=nb_dims) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [data_collate(samples, nb_dims=nb_dims) for samples in transposed]  # Backwards compatibility.
        else:
            try:
                return elem_type([data_collate(samples, nb_dims=nb_dims) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [data_collate(samples, nb_dims=nb_dims) for samples in transposed]

    raise TypeError(data_collate_err_msg_format.format(elem_type))

def move_tensor(tensor, device):
    """
    Move minibatch tensor to CPU/GPU.
    """
    if isinstance(tensor, list):
        return [move_tensor(t, device) if t is not None else None for t in tensor]
    else:
        return tensor.to(device)

def move_tensors(*tensors, device):
    """
    Move minibatch tensors to CPU/GPU.
    """
    return [move_tensor(tensor, device) if tensor is not None else None for tensor in tensors]

def concat_tensors(features):
    f = []
    for feature in features:
        if isinstance(feature, list) or isinstance(feature, tuple):
            f = f + [*feature]
        else:
            f = f + [feature]
    return torch.cat(f, dim=-1)


class SinusoidalEncoder(nn.Module):
    def __init__(self, data=None, dim=None, max_period=10000, dim_scale=0.25):
        super().__init__()
        assert data is not None or dim is not None

        if dim is None:
            dim = int(len(data)**dim_scale)

        half = dim // 2
        self.freqs = nn.Parameter(
            torch.exp(-math.log(max_period) * 
                torch.arange(start=0, end=half, dtype=torch.float) / half
            ), requires_grad=False)

        self.res = dim % 2

    def forward(self, timesteps):
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.FloatTensor(timesteps)
        timesteps = timesteps.to(self.freqs.device)

        args = (timesteps[..., None] * self.freqs)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return torch.cat([embedding, torch.zeros_like(embedding[..., :self.res])], dim=-1)


class OneHotEncoder(nn.Module):
    def __init__(self, data=None, dim=None):
        super().__init__()
        assert data is not None or dim is not None

        self.transform = None
        if data is not None:
            self.transform = preprocessing.LabelEncoder().fit(data)
        if dim is None:
            dim = len(np.unique(self.transform.transform(data)))

        self.eye = nn.Parameter(torch.eye(dim), requires_grad=False)

    def forward(self, labels):
        if self.transform:
            labels = self.transform.transform(labels)

        return self.eye[labels]


class AttrEncoder(nn.Module):
    def __init__(self, data: np.ndarray,
                 discrete_dim: int = None, continuous_dim: int = None):
        super().__init__()

        encoder = []
        sample = data[0].tolist()
        for i, s in enumerate(sample):
            if type(s) in (bool, str):
                encoder.append(OneHotEncoder(
                    data=data[:, i], dim=discrete_dim))
            else:
                encoder.append(SinusoidalEncoder(
                    data=data[:, i], dim=continuous_dim))

        self.encoder = nn.Sequential(*encoder)

    def forward(self, data):
        if data.ndim < 2:
            out = data[None, :]
        else:
            out = data

        out = [enc(np.array(out[:, i].tolist())) for i, enc in enumerate(self.encoder)]

        out = torch.cat(out, dim=-1)
        if out.ndim > data.ndim:
            out = out.squeeze(-2)
        return out
