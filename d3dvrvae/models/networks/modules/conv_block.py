from torch import Tensor, nn, cat
from torch.nn.functional import group_norm, leaky_relu


IdenticalConvBlockConvParams = {
    "kernel_size": 3,
    "stride": 1,
    "padding": 1,
    "output_padding": 0,
}


class ConvBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int | None = None,
        transpose: bool = False,
        act_norm: bool = True,
    ) -> None:
        super().__init__()
        self.act_norm = act_norm

        if not transpose:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        else:
            if output_padding is None:
                output_padding = stride // 2
            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
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
            transpose=transpose,
            act_norm=act_norm,
            **IdenticalConvBlockConvParams,
        )


class ConvBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int | None = None,
        transpose: bool = False,
        act_norm: bool = True,
    ) -> None:
        super().__init__()
        self.act_norm = act_norm

        if not transpose:
            self.conv = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        else:
            if output_padding is None:
                output_padding = stride // 2
            self.conv = nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
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
            transpose=transpose,
            act_norm=act_norm,
            **IdenticalConvBlockConvParams,
        )


class ConvModuleBase(nn.Module):
    layers: nn.ModuleList
    use_skip: bool
    debug_show_dim: bool

    def _forward(self, x: Tensor, hs: list[Tensor] | None = None) -> tuple[Tensor, list[Tensor]]:
        if self.use_skip:
            assert hs is not None and len(hs) == len(self.layers), f"{len(hs)} != {len(self.layers)}"
        else:
            assert hs is None

        xs = []
        for i, layer in enumerate(self.layers):
            if self.use_skip:
                x = cat([x, hs[i]], dim=1)
            x = layer(x)
            if self.debug_show_dim:
                print(f"{self.__class__.__name__} Layer {i}", x.shape)
            xs.append(x)

        return x, xs[:-1]


class ConvModule2d(ConvModuleBase):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            latent_dim: int,
            conv_params: list[dict[str, int]],
            transpose: bool,
            debug_show_dim: bool = False,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for i, conv_param in conv_params[:-1]:
            self.layers.append(nn.Sequential(
                ConvBlock2d(
                    latent_dim if i > 0 else in_channels,
                    latent_dim,
                    transpose=transpose,
                    **conv_param,
                ),
                IdenticalConvBlock2d(
                    latent_dim,
                    latent_dim,
                    transpose=False,
                ),
            ))
        self.layers.append(ConvBlock2d(
            latent_dim,
            out_channels,
            transpose=transpose,
            act_norm=False,
            **conv_params[-1],
        ))

        self.debug_show_dim = debug_show_dim
        self.use_skip = False

    def forward(self, x: Tensor) -> Tensor:
        return self._forward(x)[0]


class ConvModule3d(ConvModuleBase):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            latent_dim: int,
            conv_params: list[dict[str, int]],
            transpose: bool,
            debug_show_dim: bool = False,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for i, conv_param in conv_params[:-1]:
            self.layers.append(nn.Sequential(
                ConvBlock2d(
                    latent_dim if i > 0 else in_channels,
                    latent_dim,
                    transpose=transpose,
                    **conv_param,
                ),
                IdenticalConvBlock2d(
                    latent_dim,
                    latent_dim,
                    transpose=False,
                ),
            ))
        self.layers.append(ConvBlock2d(
            latent_dim,
            out_channels,
            transpose=transpose,
            act_norm=False,
            **conv_params[-1],
        ))

        self.debug_show_dim = debug_show_dim
        self.use_skip = False

    def forward(self, x: Tensor) -> Tensor:
        return self._forward(x)[0]
