from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .option import LossOption


@dataclass
class MStdLossOption(LossOption):
    pass


def create_mstd_loss() -> nn.Module:
    return MStdLoss()


class MStdLoss(nn.Module):
    @property
    def required_kwargs(self) -> list[str]:
        return ["latent"]

    def forward(self, input: Tensor, target: Tensor, latent: list[Tensor]) -> Tensor:
        return sum([torch.sqrt(torch.mean(v**2)) for v in latent])  # type: ignore
