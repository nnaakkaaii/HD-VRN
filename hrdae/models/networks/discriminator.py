from dataclasses import dataclass, field

from torch import nn, Tensor

from .modules import ConvModule2d, ConvModule3d, PixelWiseConv2d, PixelWiseConv3d
from .option import NetworkOption


@dataclass
class Discriminator2dOption(NetworkOption):
    hidden_channels: int = 64
    conv_params: list[dict[str, list[int]]] = field(
        default_factory=lambda: [
            {"kernel_size": [3], "stride": [2], "padding": [1], "output_padding": [1]},
        ]
    )
    debug_show_dim: bool = False


def create_discriminator2d(out_channels: int, opt: Discriminator2dOption) -> nn.Module:
    return Discriminator2d(
        in_channels=out_channels,
        out_channels=out_channels,
        hidden_channels=opt.hidden_channels,
        conv_params=opt.conv_params,
        debug_show_dim=opt.debug_show_dim,
    )


class Discriminator2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        conv_params: list[dict[str, list[int]]],
        debug_show_dim: bool,
    ) -> None:
        super().__init__()
        self.cnn = ConvModule2d(
            in_channels,
            out_channels,
            hidden_channels,
            conv_params,
            transpose=False,
            act_norm=False,
            debug_show_dim=debug_show_dim,
        )
        self.bottleneck = PixelWiseConv2d(
            hidden_channels,
            out_channels,
            act_norm=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.cnn(x)
        z = self.bottleneck(y)
        return z.mean(2).mean(2)


@dataclass
class Discriminator3dOption(NetworkOption):
    hidden_channels: int = 64
    conv_params: list[dict[str, list[int]]] = field(
        default_factory=lambda: [
            {"kernel_size": [3], "stride": [2], "padding": [1], "output_padding": [1]},
        ]
    )
    debug_show_dim: bool = False


def create_discriminator3d(out_channels: int, opt: Discriminator3dOption) -> nn.Module:
    return Discriminator3d(
        in_channels=out_channels,
        out_channels=out_channels,
        hidden_channels=opt.hidden_channels,
        conv_params=opt.conv_params,
        debug_show_dim=opt.debug_show_dim,
    )


class Discriminator3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        conv_params: list[dict[str, list[int]]],
        debug_show_dim: bool,
    ) -> None:
        super().__init__()
        self.cnn = ConvModule3d(
            in_channels,
            hidden_channels,
            hidden_channels,
            conv_params,
            transpose=False,
            act_norm=True,
            debug_show_dim=debug_show_dim,
        )
        self.bottleneck = PixelWiseConv3d(
            hidden_channels,
            out_channels,
            act_norm=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.cnn(x)
        z = self.bottleneck(y)
        return z.mean(2).mean(2).mean(2)
