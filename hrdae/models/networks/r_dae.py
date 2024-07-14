# Licensed under the Apache License, Version 2.0 (the "License");
# Recurrent Disentangled AutoEncoder (R-DAE)

from dataclasses import dataclass

from torch import Tensor, nn

from .autoencoder import AEDecoder2d, AEDecoder3d, AEEncoder2d, AEEncoder3d
from .functions import upsample_motion_tensor
from .modules import (
    IdenticalConvBlockConvParams,
    create_activation,
    create_aggregator2d,
    create_aggregator3d,
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
    aggregator: str = "attention"
    cycle: bool = False


@dataclass
class RDAE3dOption(RAE3dOption):
    in_channels: int = 1  # 2 if content_phase = "all"
    aggregator: str = "attention"
    cycle: bool = False


def create_rdae2d(out_channels: int, opt: RDAE2dOption) -> nn.Module:
    motion_encoder = create_motion_encoder1d(
        opt.latent_dim, opt.debug_show_dim, opt.motion_encoder
    )
    if opt.cycle:
        return CycleRDAE2d(
            opt.in_channels,
            out_channels,
            opt.hidden_channels,
            opt.latent_dim,
            opt.conv_params,
            motion_encoder,
            opt.activation,
            opt.aggregator,
            opt.debug_show_dim,
        )
    return RDAE2d(
        opt.in_channels,
        out_channels,
        opt.hidden_channels,
        opt.latent_dim,
        opt.conv_params,
        motion_encoder,
        opt.activation,
        opt.aggregator,
        opt.debug_show_dim,
    )


def create_rdae3d(out_channels: int, opt: RDAE3dOption) -> nn.Module:
    motion_encoder = create_motion_encoder2d(
        opt.latent_dim, opt.debug_show_dim, opt.motion_encoder
    )
    if opt.cycle:
        return CycleRDAE3d(
            opt.in_channels,
            out_channels,
            opt.hidden_channels,
            opt.latent_dim,
            opt.conv_params,
            motion_encoder,
            opt.activation,
            opt.aggregator,
            opt.debug_show_dim,
        )
    return RDAE3d(
        opt.in_channels,
        out_channels,
        opt.hidden_channels,
        opt.latent_dim,
        opt.conv_params,
        motion_encoder,
        opt.activation,
        opt.aggregator,
        opt.debug_show_dim,
    )


class RDAE2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        motion_encoder: MotionEncoder1d,
        activation: str,
        aggregator: str,
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()

        self.content_encoder = AEEncoder2d(
            in_channels,
            hidden_channels,
            latent_dim,
            conv_params + [IdenticalConvBlockConvParams],
            debug_show_dim,
        )
        self.motion_encoder = motion_encoder
        self.decoder = AEDecoder2d(
            out_channels,
            hidden_channels,
            2 * latent_dim if aggregator == "concatenation" else latent_dim,
            conv_params[::-1],
            debug_show_dim,
        )
        self.activation = create_activation(activation)
        self.aggregator = create_aggregator2d(aggregator, latent_dim, latent_dim)

    def forward(
        self,
        x_1d: Tensor,
        x_2d_0: Tensor,
        x_1d_0: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor], Tensor, list[Tensor]]:
        c, _ = self.content_encoder(x_2d_0)
        m = self.motion_encoder(x_1d, x_1d_0)
        b, t, c_, h_, w = m.size()
        m_reshaped = m.reshape(b * t, c_, h_, w)
        c_exp = c.repeat(t, 1, 1, 1)
        m_reshaped = upsample_motion_tensor(m_reshaped, c_exp)
        h = self.aggregator((m_reshaped, c_exp))
        y = self.decoder(h)
        _, c_, h_, w = y.size()
        y = y.reshape(b, t, c_, h_, w)
        if self.activation is not None:
            y = self.activation(y)
        return y, [c], m, []


class CycleRDAE2d(RDAE2d):
    def forward(
        self,
        x_1d: Tensor,
        x_2d_0: Tensor,
        x_1d_0: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor], Tensor, list[Tensor]]:
        y, cs, m, _ = super().forward(x_1d, x_2d_0, x_1d_0)
        b, t, c, h, w = y.size()
        y_seq = y.reshape(b * t, c, h, w)
        d, _ = self.content_encoder(y_seq)
        assert len(cs) == 1
        assert d.size(0) == b * t
        d = d.reshape(b, t, *d.size()[1:])
        return y, [c.unsqueeze(1) for c in cs], m, [d]


class RDAE3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        motion_encoder: MotionEncoder2d,
        activation: str,
        aggregator: str,
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()
        self.content_encoder = AEEncoder3d(
            in_channels,
            hidden_channels,
            latent_dim,
            conv_params + [IdenticalConvBlockConvParams],
            debug_show_dim,
        )
        self.motion_encoder = motion_encoder
        self.decoder = AEDecoder3d(
            out_channels,
            hidden_channels,
            2 * latent_dim if aggregator == "concatenation" else latent_dim,
            conv_params[::-1],
            debug_show_dim,
        )
        self.activation = create_activation(activation)
        self.aggregator = create_aggregator3d(aggregator, latent_dim, latent_dim)

    def forward(
        self,
        x_2d: Tensor,
        x_3d_0: Tensor,
        x_2d_0: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor], Tensor, list[Tensor]]:
        c, _ = self.content_encoder(x_3d_0)
        m = self.motion_encoder(x_2d, x_2d_0)
        b, t, c_, d, h_, w = m.size()
        m_reshaped = m.reshape(b * t, c_, d, h_, w)
        c_exp = c.repeat(t, 1, 1, 1, 1)
        m_reshaped = upsample_motion_tensor(m_reshaped, c_exp)
        h = self.aggregator((m_reshaped, c_exp))
        y = self.decoder(h)
        _, c_, d, h_, w = y.size()
        y = y.reshape(b, t, c_, d, h_, w)
        if self.activation is not None:
            y = self.activation(y)
        return y, [c], m, []


class CycleRDAE3d(RDAE3d):
    def forward(
        self,
        x_2d: Tensor,
        x_3d_0: Tensor,
        x_2d_0: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor], Tensor, list[Tensor]]:
        y, cs, m, _ = super().forward(x_2d, x_3d_0, x_2d_0)
        b, t, c, d_, h, w = y.size()
        y_seq = y.reshape(b * t, c, d_, h, w)
        d, _ = self.content_encoder(y_seq)
        assert len(cs) == 1
        assert d.size(0) == b * t
        d = d.reshape(b, t, *d.size()[1:])
        return y, [c.unsqueeze(1) for c in cs], m, [d]
