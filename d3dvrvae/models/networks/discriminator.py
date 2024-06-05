from dataclasses import dataclass, field

import torch
from torch import nn

from .modules import ConvEncoder3d


@dataclass
class Discriminator3d:
    in_channels: int = 1
    latent_dim: int = 64
    conv_params: list[dict[str, int]] = field(
        default_factory=lambda: [
            {"kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
            {"kernel_size": 3, "stride": 1, "padding": 1, "output_padding": 0},
        ]
    )
    debug_show_dim: bool = False


def create_discriminator_3d(opt: Discriminator3d) -> nn.Module:
    return ConvEncoder3d(
        opt.in_channels,
        opt.latent_dim,
        opt.conv_params,
        opt.debug_show_dim,
    )


class Discriminator3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, int]],
        debug_show_dim: bool,
    ) -> None:
        super().__init__()

        self.encoder = ConvEncoder3d(in_channels, latent_dim, conv_params, debug_show_dim)
        self.readout = nn.Conv3d(latent_dim, 2, 1)  # 2 classes

    def forward(self, x):
        h = self.encoder(x)
        y = self.readout(h)
        return y
