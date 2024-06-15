# Licensed under the Apache License, Version 2.0 (the "License");
# Recurrent AutoEncoder

from dataclasses import dataclass, field

from omegaconf import MISSING
from torch import Tensor, nn
from torch.nn.functional import interpolate

from .modules import ConvModule2d, ConvModule3d, create_activation
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
class RAE2dOption(NetworkOption):
    out_channels: int = 1
    latent_dim: int = 64
    conv_params: list[dict[str, list[int]]] = field(
        default_factory=lambda: [{"kernel_size": [3], "stride": [2], "padding": [1]}]
        * 3,
    )
    motion_encoder: MotionEncoder1dOption = MISSING
    upsample_size: list[int] = field(default_factory=lambda: [8, 8])
    debug_show_dim: bool = False


@dataclass
class RAE3dOption(NetworkOption):
    out_channels: int = 1
    latent_dim: int = 64
    conv_params: list[dict[str, list[int]]] = field(
        default_factory=lambda: [{"kernel_size": [3], "stride": [2], "padding": [1]}]
        * 3,
    )
    motion_encoder: MotionEncoder2dOption = MISSING
    upsample_size: list[int] = field(default_factory=lambda: [8, 8, 8])
    debug_show_dim: bool = False


def create_rae2d(opt: RAE2dOption) -> nn.Module:
    motion_encoder = create_motion_encoder1d(
        opt.latent_dim, opt.debug_show_dim, opt.motion_encoder
    )
    return RAE2d(
        opt.out_channels,
        opt.latent_dim,
        opt.conv_params,
        motion_encoder,
        opt.upsample_size,
        opt.activation,
        opt.debug_show_dim,
    )


def create_rae3d(opt: RAE3dOption) -> nn.Module:
    motion_encoder = create_motion_encoder2d(
        opt.latent_dim, opt.debug_show_dim, opt.motion_encoder
    )
    return RAE3d(
        opt.out_channels,
        opt.latent_dim,
        opt.conv_params,
        motion_encoder,
        opt.upsample_size,
        opt.activation,
        opt.debug_show_dim,
    )


class Decoder2d(nn.Module):
    def __init__(
        self,
        out_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()
        self.dec = ConvModule2d(
            latent_dim,
            out_channels,
            latent_dim,
            conv_params,
            transpose=True,
            debug_show_dim=debug_show_dim,
        )

    def forward(self, m: Tensor) -> Tensor:
        return self.dec(m)


class Decoder3d(nn.Module):
    def __init__(
        self,
        out_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()
        self.dec = ConvModule3d(
            latent_dim,
            out_channels,
            latent_dim,
            conv_params,
            transpose=True,
            debug_show_dim=debug_show_dim,
        )

    def forward(self, m: Tensor) -> Tensor:
        return self.dec(m)


class RAE2d(nn.Module):
    def __init__(
        self,
        out_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        motion_encoder: MotionEncoder1d,
        upsample_size: list[int],
        activation: str,
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()
        self.motion_encoder = motion_encoder
        self.decoder = Decoder2d(
            out_channels,
            latent_dim,
            conv_params[::-1],
            debug_show_dim,
        )
        self.upsample_size = upsample_size
        self.activation = create_activation(activation)

    def forward(
        self,
        x_1d: Tensor,
        x_2d_0: Tensor | None = None,
        x_1d_0: Tensor | None = None,
    ) -> Tensor:
        m = self.motion_encoder(x_1d, x_1d_0)
        b, t, c_, h = m.size()
        m = m.reshape(b * t, c_, h, 1)
        m = interpolate(m, size=self.upsample_size, mode="bilinear", align_corners=True)
        y = self.decoder(m)
        _, c_, h, w = y.size()
        y = y.reshape(b, t, c_, h, w)
        if self.activation is not None:
            y = self.activation(y)
        return y


class RAE3d(nn.Module):
    def __init__(
        self,
        out_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        motion_encoder: MotionEncoder2d,
        upsample_size: list[int],
        activation: str,
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()
        self.motion_encoder = motion_encoder
        self.decoder = Decoder3d(
            out_channels,
            latent_dim,
            conv_params[::-1],
            debug_show_dim,
        )
        self.upsample_size = upsample_size
        self.activation = create_activation(activation)

    def forward(
        self,
        x_2d: Tensor,
        x_3d_0: Tensor | None = None,
        x_2d_0: Tensor | None = None,
    ) -> Tensor:
        m = self.motion_encoder(x_2d, x_2d_0)
        b, t, c_, d, h = m.size()
        m = m.view(b * t, c_, d, h, 1)
        m = interpolate(
            m, size=self.upsample_size, mode="trilinear", align_corners=True
        )
        y = self.decoder(m)
        _, c_, d, h, w = y.size()
        y = y.view(b, t, c_, d, h, w)
        if self.activation is not None:
            y = self.activation(y)
        return y
