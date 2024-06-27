from torch import Tensor, nn

from .conv_block import (
    ConvBlock1d,
    ConvBlock2d,
    ConvBlock3d,
    ConvModuleBase,
    IdenticalConvBlock1d,
    IdenticalConvBlock2d,
    IdenticalConvBlock3d,
)


class HierarchicalConvDecoder1d(ConvModuleBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        aggregation: str,
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()
        assert len(conv_params) > 1

        d = 2 if aggregation == "concatenation" else 1

        self.layers = nn.ModuleList()
        for i, conv_param in enumerate(conv_params[:-1]):
            self.layers.append(
                nn.Sequential(
                    ConvBlock1d(
                        d * latent_dim if i > 0 else d * in_channels,
                        latent_dim,  # 2*latent_dim ?
                        kernel_size=conv_param["kernel_size"],
                        stride=conv_param["stride"],
                        padding=conv_param["padding"],
                        output_padding=conv_param.get("output_padding"),
                        transpose=True,
                    ),
                    IdenticalConvBlock1d(
                        latent_dim,  # 2*latent_dim ?
                        latent_dim,
                        transpose=False,
                    ),
                )
            )
        self.layers.append(
            ConvBlock1d(
                d * latent_dim,
                out_channels,
                kernel_size=conv_params[-1]["kernel_size"],
                stride=conv_params[-1]["stride"],
                padding=conv_params[-1]["padding"],
                output_padding=conv_params[-1].get("output_padding"),
                transpose=True,
                act_norm=False,
            )
        )

        self.debug_show_dim = debug_show_dim
        self.use_skip = True
        self.aggregation = aggregation

    def forward(self, x: Tensor, hs: list[Tensor]) -> Tensor:
        return self._forward(x, hs)[0]


class HierarchicalConvDecoder2d(ConvModuleBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        aggregation: str,
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()
        assert len(conv_params) > 1

        d = 2 if aggregation == "concatenation" else 1

        self.layers = nn.ModuleList()
        for i, conv_param in enumerate(conv_params[:-1]):
            self.layers.append(
                nn.Sequential(
                    ConvBlock2d(
                        d * latent_dim if i > 0 else d * in_channels,
                        latent_dim,  # 2*latent_dim ?
                        kernel_size=conv_param["kernel_size"],
                        stride=conv_param["stride"],
                        padding=conv_param["padding"],
                        output_padding=conv_param.get("output_padding"),
                        transpose=True,
                    ),
                    IdenticalConvBlock2d(
                        latent_dim,  # 2*latent_dim ?
                        latent_dim,
                        transpose=False,
                    ),
                )
            )
        self.layers.append(
            ConvBlock2d(
                d * latent_dim,
                out_channels,
                kernel_size=conv_params[-1]["kernel_size"],
                stride=conv_params[-1]["stride"],
                padding=conv_params[-1]["padding"],
                output_padding=conv_params[-1].get("output_padding"),
                transpose=True,
                act_norm=False,
            )
        )

        self.debug_show_dim = debug_show_dim
        self.use_skip = True
        self.aggregation = aggregation

    def forward(self, x: Tensor, hs: list[Tensor]) -> Tensor:
        return self._forward(x, hs)[0]


class HierarchicalConvDecoder3d(ConvModuleBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        aggregation: str,
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()
        assert len(conv_params) > 1

        d = 2 if aggregation == "concatenation" else 1

        self.layers = nn.ModuleList()
        for i, conv_param in enumerate(conv_params[:-1]):
            self.layers.append(
                nn.Sequential(
                    ConvBlock3d(
                        d * latent_dim if i > 0 else d * in_channels,
                        latent_dim,  # 2*latent_dim ?
                        kernel_size=conv_param["kernel_size"],
                        stride=conv_param["stride"],
                        padding=conv_param["padding"],
                        output_padding=conv_param.get("output_padding"),
                        transpose=True,
                    ),
                    IdenticalConvBlock3d(
                        latent_dim,  # 2*latent_dim ?
                        latent_dim,
                        transpose=False,
                    ),
                )
            )
        self.layers.append(
            ConvBlock3d(
                d * latent_dim,
                out_channels,
                kernel_size=conv_params[-1]["kernel_size"],
                stride=conv_params[-1]["stride"],
                padding=conv_params[-1]["padding"],
                output_padding=conv_params[-1].get("output_padding"),
                transpose=True,
                act_norm=False,
            )
        )

        self.debug_show_dim = debug_show_dim
        self.use_skip = True
        self.aggregation = aggregation

    def forward(self, x: Tensor, hs: list[Tensor]) -> Tensor:
        return self._forward(x, hs)[0]
