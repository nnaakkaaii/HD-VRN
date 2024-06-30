from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from omegaconf import MISSING

from .option import LossOption
from ..networks import create_network, NetworkOption


@dataclass
class PerceptualLossOption(LossOption):
    weight: Path = MISSING
    network: NetworkOption = MISSING


def create_perceptual_loss(opt: PerceptualLossOption) -> nn.Module:
    return PerceptualLoss(opt.weight, opt.network)


class PerceptualLoss(nn.Module):
    def __init__(
        self,
        weight: Path,
        network: NetworkOption,
    ) -> None:
        super().__init__()
        self.network = create_network(1, network)
        self.network.load_state_dict(torch.load(weight))
        if torch.cuda.is_available():
            self.network.to("cuda:0")
            self.network = nn.DataParallel(self.network)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        _, z_input = self.network(input)
        _, z_target = self.network(target)
        return sum(
            [  # type: ignore
                torch.sqrt(torch.mean((z_input[i] - z_target[i]) ** 2))
                for i in range(len(z_input))
            ]
        )
