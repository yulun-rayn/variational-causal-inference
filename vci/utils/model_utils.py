from typing import Union, Iterable, List, Dict, Tuple, Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, inf

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]

def conv_1x1(in_width, out_width, dim=2):
    if dim == 1:
        return nn.Conv1d(in_width, out_width, 1, 1, 0)
    elif dim == 2:
        return nn.Conv2d(in_width, out_width, 1, 1, 0)
    elif dim == 3:
        return nn.Conv3d(in_width, out_width, 1, 1, 0)
    else:
        return ValueError("dim not recognized")

def conv_3x3(in_width, out_width, dim=2):
    if dim == 1:
        return nn.Conv1d(in_width, out_width, 3, 1, 1)
    elif dim == 2:
        return nn.Conv2d(in_width, out_width, 3, 1, 1)
    elif dim == 3:
        return nn.Conv3d(in_width, out_width, 3, 1, 1)
    else:
        return ValueError("dim not recognized")

def parse_block_string(res, width, depth, in_size=None, out_size=None):
    res = str(res).split(',')
    res = [r.split('*') for r in res]
    width = str(width).split(',')
    depth = str(depth).split(',')

    if in_size is None:
        in_size = (int(width[0]), *[int(s) for s in res[0]])

    layers_res = []
    layers_width = []
    for r, w, d in zip(res, width, depth):
        r = [int(s) for s in r]
        w, d = int(w), int(d)
        e = math.ceil(w/4)

        layers_res.extend([r] * d)
        layers_width.extend([(in_size[0], e, w)] + [(w, e, w)] * (d-1))

        in_size = (w, *r)

    if out_size is not None:
        layers_res = layers_res + [out_size[1:]]
        layers_width = layers_width + [(w, math.ceil(w/4), out_size[0])]

    return layers_res, layers_width

def lr_lambda_exp(decay_epochs, decay_rate=0.1):
    def f(epoch):
        return decay_rate ** (epoch // decay_epochs)
    return f

def lr_lambda_lin(total_epochs, fixed_epochs=100):
    def f(epoch):
        if epoch <= fixed_epochs:
            rate = 1.0
        elif epoch >= total_epochs:
            rate = 0.0
        else:
            rate = (total_epochs - epoch) / (total_epochs - fixed_epochs)
        return rate
    return f

def total_grad_norm_(
        parameters: _tensor_or_tensors, norm_type: float = 2.0,
        error_if_nonfinite: bool = False) -> torch.Tensor:
    r"""Total gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    device = grads[0].device
    if norm_type == inf:
        norms = [g.detach().abs().max().to(device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]), norm_type)
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    return total_norm
