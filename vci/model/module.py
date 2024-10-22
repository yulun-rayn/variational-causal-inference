import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    A multilayer perceptron with ReLU activations and optional BatchNorm.
    """

    def __init__(self, sizes, heads=None, batch_norm=True, final_act=None):
        super().__init__()
        self.heads = heads
        if heads is not None:
            sizes[-1] = sizes[-1] * heads

        layers = []
        for i in range(len(sizes) - 1):
            layers += [
                nn.Linear(sizes[i], sizes[i+1]),
                nn.BatchNorm1d(sizes[i+1])
                if batch_norm and i < len(sizes) - 2
                else None,
                nn.ReLU()
                if i < len(sizes) - 2
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

    def forward(self, *x):
        inputs = []
        for input in x:
            if isinstance(input, list) or isinstance(input, tuple):
                inputs = inputs + [*input]
            else:
                inputs = inputs + [input]
        inputs = torch.cat(inputs, dim=-1)

        out = self.network(inputs)

        if self.heads is not None:
            out = out.view(*out.shape[:-1], -1, self.heads)
        return out


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
