from torch import Tensor, nn

from .conv_block import (ConvBlock2d, ConvBlock3d, ConvModuleBase,
                         IdenticalConvBlock2d, IdenticalConvBlock3d)


class HierarchicalConvEncoder2d(ConvModuleBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, int]],
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for i, conv_param in enumerate(conv_params[:-1]):
            self.layers.append(
                nn.Sequential(
                    ConvBlock2d(
                        latent_dim if i > 0 else in_channels,
                        latent_dim,
                        transpose=False,
                        **conv_param,
                    ),
                    IdenticalConvBlock2d(
                        latent_dim,
                        latent_dim,
                        transpose=False,
                    ),
                )
            )
        self.layers.apend(
            ConvBlock2d(
                latent_dim,
                out_channels,
                transpose=False,
                act_norm=False,
                **conv_params[-1],
            )
        )

        self.debug_show_dim = debug_show_dim
        self.use_skip = False

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        return self._forward(x)


class HierarchicalConvEncoder3d(ConvModuleBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, int]],
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for i, conv_param in enumerate(conv_params[:-1]):
            self.layers.append(
                nn.Sequential(
                    ConvBlock3d(
                        latent_dim if i > 0 else in_channels,
                        latent_dim,
                        transpose=False,
                        **conv_param,
                    ),
                    IdenticalConvBlock3d(
                        latent_dim,
                        latent_dim,
                        transpose=False,
                    ),
                )
            )
        self.layers.append(
            ConvBlock3d(
                latent_dim,
                out_channels,
                transpose=False,
                act_norm=False,
                **conv_params[-1],
            )
        )

        self.debug_show_dim = debug_show_dim
        self.use_skip = False

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        return self._forward(x)
