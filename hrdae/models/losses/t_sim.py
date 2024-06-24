from dataclasses import dataclass

from torch import nn, Tensor

from .option import LossOption


@dataclass
class TemporalSimilarityLossOption(LossOption):
    period: int = 10


def create_tsim_loss(opt: TemporalSimilarityLossOption) -> nn.Module:
    return TemporalSimilarity(opt.period)


class TemporalSimilarity(nn.Module):
    def __init__(self, period: int) -> None:
        super().__init__()
        self.period = period

    @property
    def required_kwargs(self) -> list[str]:
        return ["latent"]

    def forward(self, input: Tensor, target: Tensor, latent: Tensor) -> Tensor:
        b = latent.size(0)
        n = latent.size()[1:]
        latent = latent.reshape(b // self.period, self.period, *n)
        return ((latent - latent.mean(dim=1, keepdim=True)) ** 2).mean()
