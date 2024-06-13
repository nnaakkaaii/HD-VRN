from dataclasses import dataclass, field

from torch import nn

from .modules import ConvModule3d


@dataclass
class Discriminator3d:
    in_channels: int = 1
    latent_dim: int = 64
    conv_params: list[dict[str, int]] = field(
        default_factory=lambda: [
            {"kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
        ]
    )
    debug_show_dim: bool = False


def create_discriminator_3d(opt: Discriminator3d) -> nn.Module:
    return ConvModule3d(
        opt.in_channels,
        2,
        opt.latent_dim,
        opt.conv_params,
        transpose=False,
        debug_show_dim=opt.debug_show_dim,
    )