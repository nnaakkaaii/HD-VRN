from dataclasses import dataclass

from torch import nn, Tensor

from .option import LossOption


@dataclass
class TemporalSimilarityLossOption(LossOption):
    pass


def create_tsim_loss() -> nn.Module:
    return TemporalSimilarity()


class TemporalSimilarity(nn.Module):
    @property
    def required_kwargs(self) -> list[str]:
        return ["latent"]

    def forward(self, input: Tensor, target: Tensor, latent: Tensor) -> Tensor:
        return ((latent - latent.mean(dim=1, keepdim=True)) ** 2).mean()
