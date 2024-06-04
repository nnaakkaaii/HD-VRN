from torch import Tensor, nn

from .conv_block import ConvBlock2d, ConvBlock3d


class ConvDecoder2d(nn.Module):
    def __init__(
        self,
        out_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, int]],
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()

        self.dec = nn.Sequential(
            *[
                ConvBlock2d(latent_dim, latent_dim, transpose=True, **p)
                for p in conv_params[::-1]
            ],
        )
        self.readout = nn.Conv2d(
            latent_dim,
            out_channels,
            1,
        )

        self.__debug_show_dim = debug_show_dim

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.dec, start=1):
            x = layer(x)
            if self.__debug_show_dim:
                print(f"{self.__class__.__name__} Layer {i}", x.shape)

        y = self.readout(x)
        if self.__debug_show_dim:
            print(f"{self.__class__.__name__} Layer {1+len(self.dec)}", y.shape)

        return y


class ConvDecoder3d(nn.Module):
    def __init__(
        self,
        out_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, int]],
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()

        self.dec = nn.Sequential(
            *[
                ConvBlock3d(latent_dim, latent_dim, transpose=True, **p)
                for p in conv_params[::-1]
            ],
        )
        self.readout = nn.Conv3d(
            latent_dim,
            out_channels,
            1,
        )

        self.__debug_show_dim = debug_show_dim

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.dec, start=1):
            x = layer(x)
            if self.__debug_show_dim:
                print(f"{self.__class__.__name__} Layer {i}", x.shape)

        y = self.readout(x)
        if self.__debug_show_dim:
            print(f"{self.__class__.__name__} Layer {1+len(self.dec)}", y.shape)

        return y
