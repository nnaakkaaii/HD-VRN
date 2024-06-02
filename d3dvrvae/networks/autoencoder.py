from dataclasses import dataclass, field

from torch import nn, Tensor, cat

from .modules import ConvSC2d
from .functions import stride_generator
from .option import NetworkOption


class Encoder2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: list[int],
                 debug_show_dim: bool = False,
                 ) -> None:
        super().__init__()

        strides = stride_generator(len(hidden_channels))
        self.enc = nn.Sequential(
            ConvSC2d(in_channels,
                     hidden_channels[0],
                     stride=strides[0],
                     ),
            *[
                ConvSC2d(h1,
                         h2,
                         stride=s,
                         )
                for h1, h2, s in zip(
                    hidden_channels[:-1],
                    hidden_channels[1:],
                    strides[1:],
                )
            ],
            )

        self.__debug_show_dim = debug_show_dim

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        enc1 = self.enc[0](x)
        latent = enc1
        for i, layer in enumerate(self.enc[1:]):
            if self.__debug_show_dim:
                print(f'{self.__class__.__name__} Layer {i}', latent.shape)
            latent = layer(latent)

        print(f'{self.__class__.__name__} Layer {len(self.enc)}', latent.shape)
        return latent, enc1


class Decoder2d(nn.Module):
    def __init__(self,
                 hidden_channels: list[int],
                 out_channel: int,
                 debug_show_dim: bool = False,
                 ) -> None:
        super().__init__()
        strides = stride_generator(len(hidden_channels), reverse=True)
        self.dec = nn.Sequential(
            *[
                ConvSC2d(
                    h1,
                    h2,
                    stride=s,
                ) for h1, h2, s in zip(
                    hidden_channels[:0:-1],
                    hidden_channels[-2::-1],
                    strides[:-1],
                )
            ],
            ConvSC2d(2*hidden_channels[0],
                     hidden_channels[0],
                     stride=strides[-1],
                     transpose=True,
                     ),
            )
        self.readout = nn.Conv2d(hidden_channels[0],
                                 out_channel,
                                 1,
                                 )

        self.__debug_show_dim = debug_show_dim

    def forward(self, hid: Tensor, enc1: Tensor) -> Tensor:
        for i, layer in enumerate(self.dec[:-1], start=1):
            hid = layer(hid)
            if self.__debug_show_dim:
                print(f'{self.__class__.__name__} Layer {i}', hid.shape)

        y = self.dec[-1](cat([hid, enc1], dim=1))
        if self.__debug_show_dim:
            print(f'{self.__class__.__name__} Layer {len(self.dec)}', y.shape)

        y = self.readout(y)
        if self.__debug_show_dim:
            print(f'{self.__class__.__name__} Layer {1+len(self.dec)}', y.shape)

        return y


@dataclass
class AutoEncoder2dNetworkOption(NetworkOption):
    in_channels: int = 1
    hidden_channels: list[int] = field(default_factory=lambda: [2, 4, 4, 8, 8])
    debug_show_dim: bool = False


def create_autoencoder2d(opt: AutoEncoder2dNetworkOption) -> 'AutoEncoder2d':
    return AutoEncoder2d(in_channels=opt.in_channels,
                         hidden_channels=opt.hidden_channels,
                         debug_show_dim=opt.debug_show_dim,
                         )


class AutoEncoder2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: list[int],
                 debug_show_dim: bool,
                 ) -> None:
        super().__init__()

        self.encoder = Encoder2d(in_channels,
                                 hidden_channels,
                                 debug_show_dim,
                                 )
        self.decoder = Decoder2d(hidden_channels,
                                 in_channels,
                                 debug_show_dim,
                                 )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        latent, enc = self.encoder(x)
        y = self.decoder(latent, enc)
        return y, latent
