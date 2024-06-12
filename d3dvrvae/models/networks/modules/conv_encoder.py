from torch import Tensor, nn

from .conv_block import ConvBlock2d, ConvBlock3d


class ConvEncoderBase(nn.Module):
    enc: nn.Sequential
    debug_show_dim: bool

    def _forward(self, x: Tensor) -> list[Tensor]:
        xs = [x]
        for i, layer in enumerate(self.enc, start=1):
            xs.append(layer(xs[-1]))
            if self.debug_show_dim:
                print(f"{self.__class__.__name__} Layer {i}", xs[-1].shape)
        return xs


class ConvHierarchicalEncoder2d(ConvEncoderBase):
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

        self.debug_show_dim = debug_show_dim

    def forward(self, x: Tensor) -> list[Tensor]:
        return self._forward(x)


class ConvEncoder2d(ConvHierarchicalEncoder2d):
    def forward(self, x: Tensor) -> Tensor:
        return self._forward(x)[-1]


class ConvHierarchicalEncoder3d(nn.Module):
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

        self.debug_show_dim = debug_show_dim

    def forward(self, x: Tensor) -> list[Tensor]:
        return self._forward(x)


class ConvEncoder3d(ConvHierarchicalEncoder3d):
    def forward(self, x: Tensor) -> Tensor:
        return self._forward(x)[-1]
