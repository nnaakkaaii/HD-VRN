from dataclasses import dataclass, field

from torch import Tensor, arange, cat, nn, randint

from .modules import ConvModule2d, ConvModule3d
from .option import NetworkOption


@dataclass
class Discriminator2dOption(NetworkOption):
    hidden_channels: int = 64
    image_size: list[int] = field(default_factory=lambda: [4, 4])
    conv_params: list[dict[str, list[int]]] = field(
        default_factory=lambda: [
            {"kernel_size": [3], "stride": [2], "padding": [1], "output_padding": [1]},
        ]
    )
    debug_show_dim: bool = False


def create_discriminator2d(out_channels: int, opt: Discriminator2dOption) -> nn.Module:
    return Discriminator2d(
        in_channels=out_channels,
        out_channels=1,
        hidden_channels=opt.hidden_channels,
        image_size=opt.image_size,
        conv_params=opt.conv_params,
        debug_show_dim=opt.debug_show_dim,
    )


class Discriminator2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        image_size: list[int],
        conv_params: list[dict[str, list[int]]],
        debug_show_dim: bool,
    ) -> None:
        super().__init__()
        self.cnn = ConvModule2d(
            in_channels,
            hidden_channels,
            hidden_channels,
            conv_params,
            transpose=False,
            act_norm=False,
            debug_show_dim=debug_show_dim,
        )
        size = image_size[0] * image_size[1]
        self.bottleneck = nn.Sequential(
            nn.Linear(size * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, y: Tensor, xp: Tensor) -> Tensor:
        b, n = y.size()[:2]
        idx_y = randint(0, n, (b,))
        idx_xp = randint(0, n, (b,))
        x = cat(
            [
                y[arange(b), idx_y],
                xp[arange(b), idx_xp],
            ],
            dim=1,
        )
        h = self.cnn(x)
        z = self.bottleneck(h.reshape(b, -1))
        return z


@dataclass
class Discriminator3dOption(NetworkOption):
    hidden_channels: int = 64
    image_size: list[int] = field(default_factory=lambda: [4, 4, 4])
    conv_params: list[dict[str, list[int]]] = field(
        default_factory=lambda: [
            {"kernel_size": [3], "stride": [2], "padding": [1], "output_padding": [1]},
        ]
    )
    debug_show_dim: bool = False


def create_discriminator3d(out_channels: int, opt: Discriminator3dOption) -> nn.Module:
    return Discriminator3d(
        in_channels=out_channels,
        out_channels=1,
        hidden_channels=opt.hidden_channels,
        image_size=opt.image_size,
        conv_params=opt.conv_params,
        debug_show_dim=opt.debug_show_dim,
    )


class Discriminator3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        image_size: list[int],
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
        size = image_size[0] * image_size[1] * image_size[2]
        self.bottleneck = nn.Sequential(
            nn.Linear(size * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, y: Tensor, xp: Tensor) -> Tensor:
        b, n = y.size()[:2]
        idx_y = randint(0, n, (b,))
        idx_xp = randint(0, n, (b,))
        x = cat(
            [
                y[arange(b), idx_y],
                xp[arange(b), idx_xp],
            ],
            dim=1,
        )
        h = self.cnn(x)
        z = self.bottleneck(h.reshape(b, -1))
        return z
