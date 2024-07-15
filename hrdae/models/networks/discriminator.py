from dataclasses import dataclass, field

from torch import Tensor, nn

from .modules import ConvModule3d
from .option import NetworkOption


@dataclass
class Discriminator2dOption(NetworkOption):
    in_channels: int = 8
    hidden_channels: int = 256
    image_size: list[int] = field(default_factory=lambda: [4, 4])
    dropout_rate: float = 0.5
    fc_layer: int = 3


def create_discriminator2d(opt: Discriminator2dOption) -> nn.Module:
    return Discriminator2d(
        in_channels=opt.in_channels,
        out_channels=1,
        hidden_channels=opt.hidden_channels,
        image_size=opt.image_size,
        dropout_rate=opt.dropout_rate,
        fc_layer=opt.fc_layer,
    )


class Discriminator2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        image_size: list[int],
        fc_layer: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        size = image_size[0] * image_size[1]
        self.fc = nn.Sequential(
            nn.Linear(in_channels * size, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout1d(dropout_rate),
            *[
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.ReLU(),
                nn.Dropout1d(dropout_rate),
            ] * fc_layer,
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        z = self.fc(x.reshape(x.size(0), -1))
        return z


@dataclass
class Discriminator3dOption(NetworkOption):
    in_channels: int = 8
    hidden_channels: int = 256
    image_size: list[int] = field(default_factory=lambda: [4, 4, 4])
    conv_params: list[dict[str, list[int]]] = field(
        default_factory=lambda: [
            {"kernel_size": [3], "stride": [2], "padding": [1], "output_padding": [1]},
        ]
    )
    dropout_rate: float = 0.5
    debug_show_dim: bool = False


def create_discriminator3d(opt: Discriminator3dOption) -> nn.Module:
    return Discriminator3d(
        in_channels=opt.in_channels,
        out_channels=1,
        hidden_channels=opt.hidden_channels,
        image_size=opt.image_size,
        conv_params=opt.conv_params,
        dropout_rate=opt.dropout_rate,
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
        dropout_rate: float,
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
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout1d(dropout_rate),
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout1d(dropout_rate),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.cnn(x)
        z = self.bottleneck(h.reshape(h.size(0), -1))
        return z
