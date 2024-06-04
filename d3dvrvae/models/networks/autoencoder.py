import sys
from dataclasses import dataclass, field

from torch import nn

from .modules import ConvEncoder2d, ConvDecoder2d
from .option import NetworkOption
from ...dataloaders import Data


@dataclass
class AutoEncoder2dNetworkOption(NetworkOption):
    in_channels: int = 1
    latent_dim: int = 64
    conv_params: list[dict[str, int]] = field(
        default_factory=lambda: [
            {"kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 0},
            {"kernel_size": 3, "stride": 1, "padding": 1, "output_padding": 1},
        ]
    )
    debug_show_dim: bool = False


def create_autoencoder2d(opt: AutoEncoder2dNetworkOption) -> "AutoEncoder2d":
    return AutoEncoder2d(
        in_channels=opt.in_channels,
        latent_dim=opt.latent_dim,
        conv_params=opt.conv_params,
        debug_show_dim=opt.debug_show_dim,
    )


class AutoEncoder2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, int]],
        debug_show_dim: bool,
    ) -> None:
        super().__init__()

        self.encoder = ConvEncoder2d(
            in_channels,
            latent_dim,
            conv_params,
            debug_show_dim,
        )
        self.decoder = ConvDecoder2d(
            in_channels,
            latent_dim,
            conv_params,
            debug_show_dim,
        )

        self.__debug_show_dim = debug_show_dim

    def forward(self, x: Data) -> Data:
        y_latent = self.encoder(x.data)
        y = self.decoder(y_latent)

        if self.__debug_show_dim:
            print(f"Input", x.data.shape)
            print(f"Latent", y_latent.shape)
            print(f"Output", y.shape)
            sys.exit(0)

        return Data(y, latent=y_latent)
