# Licensed under the Apache License, Version 2.0 (the "License");
# Recurrent Disentangled AutoEncoder (R-DAE)

from dataclasses import dataclass

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
    MotionEncoder2d,
    create_motion_encoder1d,
    create_motion_encoder2d,
)
from .r_ae import RAE2dOption, RAE3dOption


@dataclass
class RDAE2dOption(RAE2dOption):
    in_channels: int = 1  # 2 if content_phase = "all"
    aggregation_method: str = "concat"


@dataclass
class RDAE3dOption(RAE3dOption):
    in_channels: int = 1  # 2 if content_phase = "all"
    aggregation_method: str = "concat"


def create_rdae2d(out_channels: int, opt: RDAE2dOption) -> nn.Module:
    motion_encoder = create_motion_encoder1d(
        opt.latent_dim, opt.debug_show_dim, opt.motion_encoder
    )
    return RDAE2d(
        opt.in_channels,
        out_channels,
        opt.latent_dim,
        opt.conv_params,
        motion_encoder,
        opt.activation,
        opt.aggregation_method,
        opt.debug_show_dim,
    )


def create_rdae3d(out_channels: int, opt: RDAE3dOption) -> nn.Module:
    motion_encoder = create_motion_encoder2d(
        opt.latent_dim, opt.debug_show_dim, opt.motion_encoder
    )
    return RDAE3d(
        opt.in_channels,
        out_channels,
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


class NormalDecoder2d(ConvModule2d):
    def __init__(
        self,
        out_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__(
            latent_dim,
            out_channels,
            latent_dim,
            conv_params,
            transpose=True,
            debug_show_dim=debug_show_dim,
        )


class NormalDecoder3d(ConvModule3d):
    def __init__(
        self,
        out_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__(
            latent_dim,
            out_channels,
            latent_dim,
            conv_params,
            transpose=True,
            debug_show_dim=debug_show_dim,
        )


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
        assert aggregation_method in ["concat", "sum"]

        self.content_encoder = NormalContentEncoder2d(
            in_channels,
            latent_dim,
            conv_params + [IdenticalConvBlockConvParams],
            debug_show_dim,
        )
        self.motion_encoder = motion_encoder
        self.decoder = NormalDecoder2d(
            out_channels,
            2 * latent_dim if aggregation_method == "concat" else latent_dim,
            conv_params[::-1],
            debug_show_dim,
        )
        self.aggregation_method = aggregation_method
        self.activation = create_activation(activation)

    def forward(
        self,
        x_1d: Tensor,
        x_2d_0: Tensor,
        x_1d_0: Tensor | None = None,
    ) -> Tensor:
        c = self.content_encoder(x_2d_0)
        m = self.motion_encoder(x_1d, x_1d_0)
        b, t, c_, h_ = m.size()
        m = m.reshape(b * t, c_, h_)
        c = c.repeat(t, 1, 1, 1)
        h = aggregate(m, c, method=self.aggregation_method)
        y = self.decoder(h)
        _, c_, h_, w = y.size()
        y = y.reshape(b, t, c_, h_, w)
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
            2 * latent_dim if aggregation_method == "concat" else latent_dim,
            conv_params[::-1],
            debug_show_dim,
        )
        self.aggregation_method = aggregation_method
        self.activation = create_activation(activation)

    def forward(
        self,
        x_2d: Tensor,
        x_3d_0: Tensor,
        x_2d_0: Tensor | None = None,
    ) -> Tensor:
        c = self.content_encoder(x_3d_0)
        m = self.motion_encoder(x_2d, x_2d_0)
        b, t, c_, d, h_ = m.size()
        m = m.reshape(b * t, c_, d, h_)
        c = c.repeat(t, 1, 1, 1, 1)
        h = aggregate(m, c, method=self.aggregation_method)
        y = self.decoder(h)
        _, c_, d, h_, w = y.size()
        y = y.reshape(b, t, c_, d, h_, w)
        if self.activation is not None:
            y = self.activation(y)
        return y
