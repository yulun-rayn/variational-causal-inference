import torch
import torch.nn as nn

class CompoundEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input: batch_size x num_index
        weight_rep = self.weight.repeat(input.shape[1],1,1)
        weight_gat = weight_rep[torch.arange(input.shape[1]), input]
        return weight_gat.sum(1)
