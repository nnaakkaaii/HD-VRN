from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import MISSING
from torch import Tensor, nn

from ..networks import AEEncoder2dNetworkOption, NetworkOption, create_network
from .option import LossOption


@dataclass
class Perceptual2dLossOption(LossOption):
    activation: str = MISSING
    in_channels: int = MISSING
    hidden_channels: int = MISSING
    latent_dim: int = MISSING
    conv_params: list[dict[str, list[int]]] = MISSING
    weight: Path = MISSING


def create_perceptual2d_loss(opt: Perceptual2dLossOption) -> nn.Module:
    return Perceptual2dLoss(
        opt.weight,
        AEEncoder2dNetworkOption(
            activation=opt.activation,
            in_channels=opt.in_channels,
            hidden_channels=opt.hidden_channels,
            latent_dim=opt.latent_dim,
            conv_params=opt.conv_params,
        ),
    )


class Perceptual2dLoss(nn.Module):
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
        b, t = input.size()[:2]
        size = input.size()[2:]
        _, z_input = self.network(input.reshape(b * t, *size))
        _, z_target = self.network(target.reshape(b * t, *size))
        return sum(
            [  # type: ignore
                torch.sqrt(torch.mean((z_input[i] - z_target[i]) ** 2))
                for i in range(len(z_input))
            ]
        )
