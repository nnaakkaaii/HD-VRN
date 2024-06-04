from torch import nn, Tensor

from .conv_block import ConvBlock2d, ConvBlock3d


class ConvEncoder2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            latent_dim: int,
            conv_params: list[dict[str, int]],
            debug_show_dim: bool = False,
    ) -> None:
        super().__init__()

        self.enc = nn.Sequential(
            ConvBlock2d(in_channels, latent_dim, **conv_params[0]),
            *[ConvBlock2d(latent_dim, latent_dim, **p) for p in conv_params[1:]],
        )

        self.__debug_show_dim = debug_show_dim

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.enc, start=1):
            x = layer(x)
            if self.__debug_show_dim:
                print(f"{self.__class__.__name__} Layer {i}", x.shape)
        return x


class ConvEncoder3d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            latent_dim: int,
            conv_params: list[dict[str, int]],
            debug_show_dim: bool = False,
    ) -> None:
        super().__init__()

        self.enc = nn.Sequential(
            ConvBlock3d(in_channels, latent_dim, **conv_params[0]),
            *[ConvBlock3d(latent_dim, latent_dim, **p) for p in conv_params[1:]],
        )

        self.__debug_show_dim = debug_show_dim

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.enc, start=1):
            x = layer(x)
            if self.__debug_show_dim:
                print(f"{self.__class__.__name__} Layer {i}", x.shape)
        return x
