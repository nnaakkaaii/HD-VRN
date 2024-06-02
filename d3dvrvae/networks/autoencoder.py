import sys
from dataclasses import dataclass, field

from torch import nn, Tensor

from .modules import ConvBlock2d
from .option import NetworkOption


class Encoder2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 conv_params: list[dict[str, int]],
                 debug_show_dim: bool = False,
                 ) -> None:
        super().__init__()

        self.enc = nn.Sequential(
            ConvBlock2d(in_channels, latent_dim, **conv_params[0]),
            *[ConvBlock2d(latent_dim, latent_dim, **p)
              for p in conv_params[1:]],
            )

        self.__debug_show_dim = debug_show_dim

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.enc, start=1):
            x = layer(x)
            if self.__debug_show_dim:
                print(f'{self.__class__.__name__} Layer {i}', x.shape)
        return x


class Decoder2d(nn.Module):
    def __init__(self,
                 out_channels: int,
                 latent_dim: int,
                 conv_params: list[dict[str, int]],
                 debug_show_dim: bool = False,
                 ) -> None:
        super().__init__()

        self.dec = nn.Sequential(
            *[ConvBlock2d(latent_dim, latent_dim, transpose=True, **p)
              for p in conv_params[::-1]],
            )
        self.readout = nn.Conv2d(latent_dim,
                                 out_channels,
                                 1,
                                 )

        self.__debug_show_dim = debug_show_dim

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.dec, start=1):
            x = layer(x)
            if self.__debug_show_dim:
                print(f'{self.__class__.__name__} Layer {i}', x.shape)

        y = self.readout(x)
        if self.__debug_show_dim:
            print(f'{self.__class__.__name__} Layer {1+len(self.dec)}', y.shape)

        return y


@dataclass
class AutoEncoder2dNetworkOption(NetworkOption):
    in_channels: int = 1
    latent_dim: int = 64
    conv_params: list[dict[str, int]] = field(default_factory=lambda: [
        {'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 0},
        {'kernel_size': 3, 'stride': 1, 'padding': 1, 'output_padding': 1},
        ])
    debug_show_dim: bool = False


def create_autoencoder2d(opt: AutoEncoder2dNetworkOption) -> 'AutoEncoder2d':
    return AutoEncoder2d(in_channels=opt.in_channels,
                         latent_dim=opt.latent_dim,
                         conv_params=opt.conv_params,
                         debug_show_dim=opt.debug_show_dim,
                         )


class AutoEncoder2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 conv_params: list[dict[str, int]],
                 debug_show_dim: bool,
                 ) -> None:
        super().__init__()

        self.encoder = Encoder2d(in_channels,
                                 latent_dim,
                                 conv_params,
                                 debug_show_dim,
                                 )
        self.decoder = Decoder2d(in_channels,
                                 latent_dim,
                                 conv_params,
                                 debug_show_dim,
                                 )
        
        self.__debug_show_dim = debug_show_dim

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.encoder(x)
        y = self.decoder(latent)
        
        if self.__debug_show_dim:
            print(f'Input', x.shape)
            print(f'Latent', latent.shape)
            print(f'Output', y.shape)
            sys.exit(0)
            
        return y, latent
