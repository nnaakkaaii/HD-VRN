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
        return ["latent", "cycled_latent"]

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        latent: list[Tensor],
        cycled_latent: list[Tensor],
    ) -> Tensor:
        assert len(latent) == len(cycled_latent)
        return sum(
            [  # type: ignore
                torch.sqrt(((v1 - v2) ** 2).mean())
                for v1, v2 in zip(latent, cycled_latent)
            ]
        )
