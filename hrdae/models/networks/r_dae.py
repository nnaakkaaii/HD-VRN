# Licensed under the Apache License, Version 2.0 (the "License");
# Recurrent Disentangled AutoEncoder (R-DAE)

from dataclasses import dataclass, field

from omegaconf import MISSING
from torch import Tensor, nn

from .functions import aggregate
from .modules import (
    ConvModule2d,
    ConvModule3d,
    IdenticalConvBlockConvParams,
    create_activation,
)
from .motion_encoder import (
    MotionEncoder1d,
    MotionEncoder1dOption,
    MotionEncoder2d,
    MotionEncoder2dOption,
    create_motion_encoder1d,
    create_motion_encoder2d,
)
from .option import NetworkOption


@dataclass
class RDAE2dOption(NetworkOption):
    in_channels: int = 2
    out_channels: int = 1
    latent_dim: int = 64
    conv_params: list[dict[str, list[int]]] = field(
        default_factory=lambda: [{"kernel_size": [3], "stride": [2], "padding": [1]}]
        * 3,
    )
    motion_encoder: MotionEncoder1dOption = MISSING
    aggregation_method: str = "concat"
    debug_show_dim: bool = False


@dataclass
class RDAE3dOption(NetworkOption):
    in_channels: int = 2
    out_channels: int = 1
    latent_dim: int = 64
    conv_params: list[dict[str, list[int]]] = field(
        default_factory=lambda: [{"kernel_size": [3], "stride": [2], "padding": [1]}]
        * 3,
    )
    motion_encoder: MotionEncoder2dOption = MISSING
    aggregation_method: str = "concat"
    debug_show_dim: bool = False


def create_rdae2d(opt: RDAE2dOption) -> nn.Module:
    motion_encoder = create_motion_encoder1d(
        opt.latent_dim, opt.debug_show_dim, opt.motion_encoder
    )
    return RDAE2d(
        opt.in_channels,
        opt.out_channels,
        opt.latent_dim,
        opt.conv_params,
        motion_encoder,
        opt.activation,
        opt.aggregation_method,
        opt.debug_show_dim,
    )


def create_rdae3d(opt: RDAE3dOption) -> nn.Module:
    motion_encoder = create_motion_encoder2d(
        opt.latent_dim, opt.debug_show_dim, opt.motion_encoder
    )
    return RDAE3d(
        opt.in_channels,
        opt.out_channels,
        opt.latent_dim,
        opt.conv_params,
        motion_encoder,
        opt.activation,
        opt.aggregation_method,
        opt.debug_show_dim,
    )


class NormalContentEncoder2d(ConvModule2d):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            latent_dim,
            latent_dim,
            conv_params,
            transpose=False,
            debug_show_dim=debug_show_dim,
        )


class NormalContentEncoder3d(ConvModule3d):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            latent_dim,
            latent_dim,
            conv_params,
            transpose=False,
            debug_show_dim=debug_show_dim,
        )


class NormalDecoder2d(nn.Module):
    def __init__(
        self,
        out_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        aggregation_method: str = "concat",
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()
        assert aggregation_method in ["concat", "sum"]

        if aggregation_method == "concat":
            latent_dim *= 2

        self.dec = ConvModule2d(
            latent_dim,
            out_channels,
            latent_dim,
            conv_params,
            transpose=True,
            debug_show_dim=debug_show_dim,
        )
        self.aggregation_method = aggregation_method

    def forward(self, m: Tensor, c: Tensor) -> Tensor:
        return self.dec(aggregate(m, c, method=self.aggregation_method))


class NormalDecoder3d(nn.Module):
    def __init__(
        self,
        out_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        aggregation_method: str = "concat",
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()
        assert aggregation_method in ["concat", "sum"]

        if aggregation_method == "concat":
            latent_dim *= 2

        self.dec = ConvModule3d(
            latent_dim,
            out_channels,
            latent_dim,
            conv_params,
            transpose=True,
            debug_show_dim=debug_show_dim,
        )
        self.aggregation_method = aggregation_method

    def forward(self, m: Tensor, c: Tensor) -> Tensor:
        return self.dec(aggregate(m, c, method=self.aggregation_method))


class RDAE2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        motion_encoder: MotionEncoder1d,
        activation: str,
        aggregation_method: str = "concat",
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()
        self.content_encoder = NormalContentEncoder2d(
            in_channels,
            latent_dim,
            conv_params + [IdenticalConvBlockConvParams],
            debug_show_dim,
        )
        self.motion_encoder = motion_encoder
        self.decoder = NormalDecoder2d(
            out_channels,
            latent_dim,
            conv_params[::-1],
            aggregation_method,
            debug_show_dim,
        )
        self.activation = create_activation(activation)

    def forward(
        self,
        x_1d: Tensor,
        x_2d_0: Tensor,
        x_1d_0: Tensor | None = None,
    ) -> Tensor:
        c = self.content_encoder(x_2d_0)
        m = self.motion_encoder(x_1d, x_1d_0)
        b, t, c_, h = m.size()
        m = m.view(b * t, c_, h)
        c = c.repeat(t, 1, 1, 1)
        y = self.decoder(m, c)
        _, c_, h, w = y.size()
        y = y.view(b, t, c_, h, w)
        if self.activation is not None:
            y = self.activation(y)
        return y


class RDAE3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        motion_encoder: MotionEncoder2d,
        activation: str,
        aggregation_method: str = "concat",
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()
        self.content_encoder = NormalContentEncoder3d(
            in_channels,
            latent_dim,
            conv_params + [IdenticalConvBlockConvParams],
            debug_show_dim,
        )
        self.motion_encoder = motion_encoder
        self.decoder = NormalDecoder3d(
            out_channels,
            latent_dim,
            conv_params[::-1],
            aggregation_method,
            debug_show_dim,
        )
        self.activation = create_activation(activation)

    def forward(
        self,
        x_2d: Tensor,
        x_3d_0: Tensor,
        x_2d_0: Tensor | None = None,
    ) -> Tensor:
        c = self.content_encoder(x_3d_0)
        m = self.motion_encoder(x_2d, x_2d_0)
        b, t, c_, d, h = m.size()
        m = m.view(b * t, c_, d, h)
        c = c.repeat(t, 1, 1, 1, 1)
        y = self.decoder(m, c)
        _, c_, d, h, w = y.size()
        y = y.view(b, t, c_, d, h, w)
        if self.activation is not None:
            y = self.activation(y)
        return y
