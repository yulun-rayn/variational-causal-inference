import math
from numbers import Real
from typing import Optional, Tuple, Union

import numpy as np

import torch
import torch.nn.functional as F

def logprob_normal(x, loc, scale, weight=None, eps=1e-8):
    var = (scale ** 2)
    log_scale = math.log(scale) if isinstance(scale, Real) else scale.log()
    res = (
        -((x - loc) ** 2) / (2 * var + eps)
        - log_scale
        - math.log(math.sqrt(2 * math.pi))
    )

    if weight is not None:
        while weight.dim() < res.dim():
            weight = weight.unsqueeze(1)
        res = res * weight
    return res

def kldiv_normal(mu1: torch.Tensor, sigma1: torch.Tensor,
        mu2: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
    logvar1 = 2 * sigma1.log()
    logvar2 = 2 * sigma2.log()

    return torch.mean(-0.5 * torch.sum(1. + logvar1-logvar2 
        - (mu1-mu2)** 2 - (logvar1-logvar2).exp(), dim = 1), dim = 0)

def logprob_bernoulli_logits(x, logit, weight=None):
    return -F.binary_cross_entropy_with_logits(logit, x, weight=weight, reduction='none')

def logprob_zinb_positive(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    pi: torch.Tensor,
    weight=None,
    eps=1e-8
):
    """
    Log likelihood (scalar) of a minibatch according to a zinb model.
    Parameters
    ----------
    x
        Data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    pi
        logit of the dropout parameter (real support) (shape: minibatch x vars)
    eps
        numerical stability constant
    Notes
    -----
    We parametrize the bernoulli using the logits, hence the softplus functions appearing.
    """
    # theta is the dispersion rate. If .ndimension() == 1, it is shared for all cells (regardless of batch or labels)
    if theta.ndimension() == 1:
        theta = theta.view(
            1, theta.size(0)
        )  # In this case, we reshape theta for broadcasting

    softplus_pi = F.softplus(-pi)  #  uses log(sigmoid(x)) = -softplus(-x)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    pi_theta_log = -pi + theta * (log_theta_eps - log_theta_mu_eps)

    case_zero = F.softplus(pi_theta_log) - softplus_pi
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (
        -softplus_pi
        + pi_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1)
    )
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    res = mul_case_zero + mul_case_non_zero

    if weight is not None:
        while weight.dim() < res.dim():
            weight = weight.unsqueeze(1)
        res = res * weight
    return res


def logprob_nb_positive(
    x: Union[torch.Tensor, np.ndarray],
    mu: Union[torch.Tensor, np.ndarray],
    theta: Union[torch.Tensor, np.ndarray],
    weight: Union[torch.Tensor, np.ndarray] = None, 
    eps: float = 1e-8,
    log_fn: callable = torch.log,
    lgamma_fn: callable = torch.lgamma,
):
    """
    Log likelihood (scalar) of a minibatch according to a nb model.
    Parameters
    ----------
    x
        data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x vars)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x vars)
    eps
        numerical stability constant
    """
    log = log_fn
    lgamma = lgamma_fn
    log_theta_mu_eps = log(theta + mu + eps)
    res = (
        theta * (log(theta + eps) - log_theta_mu_eps)
        + x * (log(mu + eps) - log_theta_mu_eps)
        + lgamma(x + theta)
        - lgamma(theta)
        - lgamma(x + 1)
    )

    if weight is not None:
        while weight.dim() < res.dim():
            weight = weight.unsqueeze(1)
        res = res * weight
    return res

def convert_mean_disp_to_counts_logits(mu, theta, eps=1e-6):
    """
    NB parameterizations conversion.
    Parameters
    ----------
    mu
        mean of the NB distribution.
    theta
        inverse overdispersion.
    eps
        constant used for numerical log stability. (Default value = 1e-6)
    Returns
    -------
    type
        the number of failures until the experiment is stopped
        and the success probability.
    """
    if not (mu is None) == (theta is None):
        raise ValueError(
            "If using the mu/theta NB parameterization, both parameters must be specified"
        )
    logits = (mu + eps).log() - (theta + eps).log()
    total_count = theta
    return total_count, logits

def convert_counts_logits_to_mean_disp(total_count, logits):
    """
    NB parameterizations conversion.
    Parameters
    ----------
    total_count
        Number of failures until the experiment is stopped.
    logits
        success logits.
    Returns
    -------
    type
        the mean and inverse overdispersion of the NB distribution.
    """
    theta = total_count
    mu = logits.exp() * theta
    return mu, theta
