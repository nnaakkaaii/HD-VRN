import sys
from dataclasses import dataclass, field

from torch import Tensor, nn

from .modules import ConvModule2d, create_activation
from .option import NetworkOption


@dataclass
class AutoEncoder2dNetworkOption(NetworkOption):
    in_channels: int = 1
    latent_dim: int = 64
    conv_params: list[dict[str, list[int]]] = field(
        default_factory=lambda: [
            {"kernel_size": [3], "stride": [2], "padding": [1], "output_padding": [1]},
        ]
    )
    debug_show_dim: bool = False


def create_autoencoder2d(opt: AutoEncoder2dNetworkOption) -> nn.Module:
    return AutoEncoder2d(
        in_channels=opt.in_channels,
        latent_dim=opt.latent_dim,
        conv_params=opt.conv_params,
        activation=opt.activation,
        debug_show_dim=opt.debug_show_dim,
    )


class AutoEncoder2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        activation: str,
        debug_show_dim: bool,
    ) -> None:
        super().__init__()

        self.encoder = ConvModule2d(
            in_channels,
            latent_dim,
            latent_dim,
            conv_params,
            transpose=False,
            debug_show_dim=debug_show_dim,
        )
        self.decoder = ConvModule2d(
            latent_dim,
            in_channels,
            latent_dim,
            conv_params,
            transpose=True,
            debug_show_dim=debug_show_dim,
        )
        self.activation = create_activation(activation)

        self.__debug_show_dim = debug_show_dim

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        y_latent = self.encoder(x)
        y = self.decoder(y_latent)
        if self.activation is not None:
            y = self.activation(y)

        if self.__debug_show_dim:
            print("Input", x.size())
            print("Latent", y_latent.size())
            print("Output", y.size())
            sys.exit(0)

        return y, y_latent
