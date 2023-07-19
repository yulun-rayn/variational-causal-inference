import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    A multilayer perceptron with ReLU activations and optional BatchNorm.
    """

    def __init__(self, sizes, heads=None, batch_norm=True, final_act=None):
        super(MLP, self).__init__()
        self.heads = heads
        if heads is not None:
            sizes[-1] = sizes[-1] * heads

        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                nn.Linear(sizes[s], sizes[s + 1]),
                nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 2
                else None,
                nn.ReLU()
                if s < len(sizes) - 2
                else None
            ]
        if final_act is None:
            pass
        elif final_act == "relu":
            layers += [nn.ReLU()]
        elif final_act == "sigmoid":
            layers += [nn.Sigmoid()]
        elif final_act == "softmax":
            layers += [nn.Softmax(dim=-1)]
        else:
            raise ValueError("final_act not recognized")

        layers = [l for l in layers if l is not None]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        out = self.network(x)

        if self.heads is not None:
            out = out.view(*out.shape[:-1], -1, self.heads)
        return out


class SinusoidalEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, max_period: int = 10000) -> None:
        super().__init__()

        emb_dim = math.ceil(embedding_dim / num_embeddings)
        half = math.ceil(emb_dim / 2)
        self.freqs = nn.Parameter(
            max_period**(-torch.arange(start=0, end=half, dtype=torch.float32) / half),
        requires_grad=False)
        self.network = MLP([2 * half * num_embeddings, embedding_dim])

    def forward(self, timesteps):
        args = (timesteps[..., None] * self.freqs).view(timesteps.shape[0], -1)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.network(embedding)


class CompoundEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((num_embeddings+1, embedding_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)
        with torch.no_grad():
            self.weight[-1].fill_(0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: batch_size x num_index
        weight_rep = self.weight.repeat(input.shape[1], 1, 1)
        weight_gat = weight_rep[torch.arange(input.shape[1]), input]
        return weight_gat.sum(1)
