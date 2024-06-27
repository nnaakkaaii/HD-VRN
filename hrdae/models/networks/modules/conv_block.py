from torch import Tensor, cat, nn
from torch.nn.functional import group_norm, leaky_relu

IdenticalConvBlockConvParams = {
    "kernel_size": [3],
    "stride": [1],
    "padding": [1],
    "output_padding": [0],
}

PixelWiseConvBlockConvParams = {
    "kernel_size": [1],
    "stride": [1],
    "padding": [0],
    "output_padding": [0],
}


def _parse_for_1d(f: list[int]) -> int:
    if len(f) == 1:
        return f[0]
    raise ValueError(f"Invalid length: {len(f)}")


def _parse_for_2d(f: list[int]) -> tuple[int, int]:
    if len(f) == 1:
        return f[0], f[0]
    if len(f) == 2:
        return f[0], f[1]
    raise ValueError(f"Invalid length: {len(f)}")


def _parse_for_3d(f: list[int]) -> tuple[int, int, int]:
    if len(f) == 1:
        return f[0], f[0], f[0]
    if len(f) == 3:
        return f[0], f[1], f[2]
    raise ValueError(f"Invalid length: {len(f)}")


class ConvBlock1d(nn.Module):
    conv: nn.Module

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: list[int],
        stride: list[int],
        padding: list[int],
        output_padding: list[int] | None = None,
        transpose: bool = False,
        act_norm: bool = True,
    ) -> None:
        super().__init__()
        self.act_norm = act_norm

        _kernel_size = _parse_for_1d(kernel_size)
        _stride = _parse_for_1d(stride)
        _padding = _parse_for_1d(padding)

        if not transpose:
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=_kernel_size,
                stride=_stride,
                padding=_padding,
            )
        else:
            _output_padding = _stride // 2
            if output_padding is not None:
                _output_padding = _parse_for_1d(output_padding)
            self.conv = nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size=_kernel_size,
                stride=_stride,
                padding=_padding,
                output_padding=_output_padding,
            )

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv(x)
        if self.act_norm:
            y = group_norm(y, 2)
            y = leaky_relu(y, 0.2, inplace=True)
        return y


class IdenticalConvBlock1d(ConvBlock1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        transpose: bool = False,
        act_norm: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=IdenticalConvBlockConvParams["kernel_size"],
            stride=IdenticalConvBlockConvParams["stride"],
            padding=IdenticalConvBlockConvParams["padding"],
            output_padding=IdenticalConvBlockConvParams.get("output_padding"),
            transpose=transpose,
            act_norm=act_norm,
        )


class PixelWiseConv1d(ConvBlock1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_norm: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=PixelWiseConvBlockConvParams["kernel_size"],
            stride=PixelWiseConvBlockConvParams["stride"],
            padding=PixelWiseConvBlockConvParams["padding"],
            output_padding=PixelWiseConvBlockConvParams.get("output_padding"),
            transpose=False,
            act_norm=act_norm,
        )


class ConvBlock2d(nn.Module):
    conv: nn.Module

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: list[int],
        stride: list[int],
        padding: list[int],
        output_padding: list[int] | None = None,
        transpose: bool = False,
        act_norm: bool = True,
    ) -> None:
        super().__init__()
        self.act_norm = act_norm

        _kernel_size = _parse_for_2d(kernel_size)
        _stride = _parse_for_2d(stride)
        _padding = _parse_for_2d(padding)

        if not transpose:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=_kernel_size,
                stride=_stride,
                padding=_padding,
            )
        else:
            _output_padding = (_stride[0] // 2, _stride[1] // 2)
            if output_padding is not None:
                _output_padding = _parse_for_2d(output_padding)
            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=_kernel_size,
                stride=_stride,
                padding=_padding,
                output_padding=_output_padding,
            )

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv(x)
        if self.act_norm:
            y = group_norm(y, 2)
            y = leaky_relu(y, 0.2, inplace=True)
        return y


class IdenticalConvBlock2d(ConvBlock2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        transpose: bool = False,
        act_norm: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=IdenticalConvBlockConvParams["kernel_size"],
            stride=IdenticalConvBlockConvParams["stride"],
            padding=IdenticalConvBlockConvParams["padding"],
            output_padding=IdenticalConvBlockConvParams.get("output_padding"),
            transpose=transpose,
            act_norm=act_norm,
        )


class PixelWiseConv2d(ConvBlock2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_norm: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=PixelWiseConvBlockConvParams["kernel_size"],
            stride=PixelWiseConvBlockConvParams["stride"],
            padding=PixelWiseConvBlockConvParams["padding"],
            output_padding=PixelWiseConvBlockConvParams.get("output_padding"),
            transpose=False,
            act_norm=act_norm,
        )


class ConvBlock3d(nn.Module):
    conv: nn.Module

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: list[int],
        stride: list[int],
        padding: list[int],
        output_padding: list[int] | None = None,
        transpose: bool = False,
        act_norm: bool = True,
    ) -> None:
        super().__init__()
        self.act_norm = act_norm

        _kernel_size = _parse_for_3d(kernel_size)
        _stride = _parse_for_3d(stride)
        _padding = _parse_for_3d(padding)

        if not transpose:
            self.conv = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=_kernel_size,
                stride=_stride,
                padding=_padding,
            )
        else:
            _output_padding = (_stride[0] // 2, _stride[1] // 2, _stride[2] // 2)
            if output_padding is not None:
                _output_padding = _parse_for_3d(output_padding)
            self.conv = nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=_kernel_size,
                stride=_stride,
                padding=_padding,
                output_padding=_output_padding,
            )

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv(x)
        if self.act_norm:
            y = group_norm(y, 2)
            y = leaky_relu(y, 0.2, inplace=True)
        return y


class IdenticalConvBlock3d(ConvBlock3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        transpose: bool = False,
        act_norm: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=IdenticalConvBlockConvParams["kernel_size"],
            stride=IdenticalConvBlockConvParams["stride"],
            padding=IdenticalConvBlockConvParams["padding"],
            output_padding=IdenticalConvBlockConvParams.get("output_padding"),
            transpose=transpose,
            act_norm=act_norm,
        )


class PixelWiseConv3d(ConvBlock3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_norm: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=PixelWiseConvBlockConvParams["kernel_size"],
            stride=PixelWiseConvBlockConvParams["stride"],
            padding=PixelWiseConvBlockConvParams["padding"],
            output_padding=PixelWiseConvBlockConvParams.get("output_padding"),
            transpose=False,
            act_norm=act_norm,
        )


class ConvModuleBase(nn.Module):
    layers: nn.ModuleList
    use_skip: bool
    debug_show_dim: bool
    aggregation: str

    def _forward(
        self, x: Tensor, hs: list[Tensor] | None = None
    ) -> tuple[Tensor, list[Tensor]]:
        if self.use_skip:
            assert hs is not None
            assert len(hs) == len(self.layers)
        else:
            assert hs is None

        xs = []
        for i, layer in enumerate(self.layers):
            if self.use_skip:
                assert hs is not None
                if self.aggregation == "concatenation":
                    x = cat([x, hs[i]], dim=1)
                elif self.aggregation == "addition":
                    x = x + hs[i]
                else:
                    raise ValueError(f"Invalid aggregation: {self.aggregation}")
            x = layer(x)
            if self.debug_show_dim:
                print(f"{self.__class__.__name__} Layer {i}", x.size())
            xs.append(x)

        return x, xs[:-1]


class ConvModule1d(ConvModuleBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        transpose: bool,
        act_norm: bool,
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()
        assert len(conv_params) > 1

        self.layers = nn.ModuleList()
        for i, conv_param in enumerate(conv_params[:-1]):
            self.layers.append(
                nn.Sequential(
                    ConvBlock1d(
                        latent_dim if i > 0 else in_channels,
                        latent_dim,
                        kernel_size=conv_param["kernel_size"],
                        stride=conv_param["stride"],
                        padding=conv_param["padding"],
                        output_padding=conv_param.get("output_padding"),
                        transpose=transpose,
                    ),
                    IdenticalConvBlock1d(
                        latent_dim,
                        latent_dim,
                        transpose=False,
                    ),
                )
            )
        self.layers.append(
            ConvBlock1d(
                latent_dim,
                out_channels,
                kernel_size=conv_params[-1]["kernel_size"],
                stride=conv_params[-1]["stride"],
                padding=conv_params[-1]["padding"],
                output_padding=conv_params[-1].get("output_padding"),
                transpose=transpose,
                act_norm=act_norm,
            )
        )

        self.debug_show_dim = debug_show_dim
        self.use_skip = False
        self.aggregation = "concatenation"

    def forward(self, x: Tensor) -> Tensor:
        return self._forward(x)[0]


class ConvModule2d(ConvModuleBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        transpose: bool,
        act_norm: bool,
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()
        assert len(conv_params) > 1

        self.layers = nn.ModuleList()
        for i, conv_param in enumerate(conv_params[:-1]):
            self.layers.append(
                nn.Sequential(
                    ConvBlock2d(
                        latent_dim if i > 0 else in_channels,
                        latent_dim,
                        kernel_size=conv_param["kernel_size"],
                        stride=conv_param["stride"],
                        padding=conv_param["padding"],
                        output_padding=conv_param.get("output_padding"),
                        transpose=transpose,
                    ),
                    IdenticalConvBlock2d(
                        latent_dim,
                        latent_dim,
                        transpose=False,
                    ),
                )
            )
        self.layers.append(
            ConvBlock2d(
                latent_dim,
                out_channels,
                kernel_size=conv_params[-1]["kernel_size"],
                stride=conv_params[-1]["stride"],
                padding=conv_params[-1]["padding"],
                output_padding=conv_params[-1].get("output_padding"),
                transpose=transpose,
                act_norm=act_norm,
            )
        )

        self.debug_show_dim = debug_show_dim
        self.use_skip = False
        self.aggregation = "concatenation"

    def forward(self, x: Tensor) -> Tensor:
        return self._forward(x)[0]


class ConvModule3d(ConvModuleBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        transpose: bool,
        act_norm: bool,
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()
        assert len(conv_params) > 1

        self.layers = nn.ModuleList()
        for i, conv_param in enumerate(conv_params[:-1]):
            self.layers.append(
                nn.Sequential(
                    ConvBlock3d(
                        latent_dim if i > 0 else in_channels,
                        latent_dim,
                        kernel_size=conv_param["kernel_size"],
                        stride=conv_param["stride"],
                        padding=conv_param["padding"],
                        output_padding=conv_param.get("output_padding"),
                        transpose=transpose,
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
                kernel_size=conv_params[-1]["kernel_size"],
                stride=conv_params[-1]["stride"],
                padding=conv_params[-1]["padding"],
                output_padding=conv_params[-1].get("output_padding"),
                transpose=transpose,
                act_norm=act_norm,
            )
        )

        self.debug_show_dim = debug_show_dim
        self.use_skip = False
        self.aggregation = "concatenation"

    def forward(self, x: Tensor) -> Tensor:
        return self._forward(x)[0]
