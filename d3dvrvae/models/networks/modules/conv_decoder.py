from torch import Tensor, nn, cat

from .conv_block import ConvBlock2d, ConvBlock3d


class ConvDecoderBase(nn.Module):
    dec: nn.Sequential
    readout: nn.Module
    debug_show_dim: bool

    def _forward(self, x: Tensor, hs: list[Tensor] | None = None) -> Tensor:
        for i, layer in enumerate(self.dec, start=1):
            if hs is not None:
                x = cat([x, hs[-i]], dim=-1)
            x = layer(x)
            if self.debug_show_dim:
                print(f"{self.__class__.__name__} Layer {i}", x.shape)

        y = self.readout(x)
        if self.debug_show_dim:
            print(f"{self.__class__.__name__} Layer {1+len(self.dec)}", y.shape)

        return y


class ConvDecoder2d(ConvDecoderBase):
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

        self.debug_show_dim = debug_show_dim

    def forward(self, x: Tensor) -> Tensor:
        return self._forward(x)


class ConvHierarchicalDecoder2d(ConvDecoderBase):
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
                ConvBlock2d(2*latent_dim, latent_dim, transpose=True, **p)
                for p in conv_params[::-1]
            ],
        )
        self.readout = nn.Conv2d(
            latent_dim,
            out_channels,
            1,
        )

        self.debug_show_dim = debug_show_dim

    def forward(self, x: Tensor, hs: list[Tensor]) -> Tensor:
        return self._forward(x, hs)


class ConvDecoder3d(ConvDecoderBase):
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

        self.debug_show_dim = debug_show_dim

    def forward(self, x: Tensor) -> Tensor:
        return self._forward(x)


class ConvHierarchicalDecoder3d(ConvDecoderBase):
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
                ConvBlock3d(2*latent_dim, latent_dim, transpose=True, **p)
                for p in conv_params[::-1]
            ],
        )
        self.readout = nn.Conv3d(
            latent_dim,
            out_channels,
            1,
        )

        self.debug_show_dim = debug_show_dim

    def forward(self, x: Tensor, hs: list[Tensor]) -> Tensor:
        return self._forward(x, hs)
