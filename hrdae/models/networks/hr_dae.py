# Licensed under the Apache License, Version 2.0 (the "License");
# Hierarchical Recurrent Disentangled AutoEncoder (HR-DAE)

from dataclasses import dataclass

from torch import Tensor, nn

from .functions import upsample_motion_tensor
from .modules import (
    HierarchicalConvDecoder2d,
    HierarchicalConvDecoder3d,
    HierarchicalConvEncoder2d,
    HierarchicalConvEncoder3d,
    IdenticalConvBlock2d,
    IdenticalConvBlock3d,
    IdenticalConvBlockConvParams,
    PixelWiseConv2d,
    PixelWiseConv3d,
    ResNetBranch,
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
from .r_dae import RDAE2dOption, RDAE3dOption


@dataclass
class HRDAE2dOption(RDAE2dOption):
    connection_aggregation: str = "concatenation"


@dataclass
class HRDAE3dOption(RDAE3dOption):
    connection_aggregation: str = "concatenation"


def create_hrdae2d(out_channels: int, opt: HRDAE2dOption) -> nn.Module:
    motion_encoder = create_motion_encoder1d(
        opt.latent_dim, opt.debug_show_dim, opt.motion_encoder
    )
    if opt.cycle:
        return CycleHRDAE2d(
            opt.in_channels,
            out_channels,
            opt.hidden_channels,
            opt.latent_dim,
            opt.conv_params,
            motion_encoder,
            opt.activation,
            opt.aggregator,
            opt.connection_aggregation,
            opt.debug_show_dim,
        )
    return HRDAE2d(
        opt.in_channels,
        out_channels,
        opt.hidden_channels,
        opt.latent_dim,
        opt.conv_params,
        motion_encoder,
        opt.activation,
        opt.aggregator,
        opt.connection_aggregation,
        opt.debug_show_dim,
    )


def create_hrdae3d(out_channels: int, opt: HRDAE3dOption) -> nn.Module:
    motion_encoder = create_motion_encoder2d(
        opt.latent_dim, opt.debug_show_dim, opt.motion_encoder
    )
    if opt.cycle:
        return CycleHRDAE3d(
            opt.in_channels,
            out_channels,
            opt.hidden_channels,
            opt.latent_dim,
            opt.conv_params,
            motion_encoder,
            opt.activation,
            opt.aggregator,
            opt.connection_aggregation,
            opt.debug_show_dim,
        )
    return HRDAE3d(
        opt.in_channels,
        out_channels,
        opt.hidden_channels,
        opt.latent_dim,
        opt.conv_params,
        motion_encoder,
        opt.activation,
        opt.aggregator,
        opt.connection_aggregation,
        opt.debug_show_dim,
    )


class HierarchicalEncoder2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()

        self.cnn = HierarchicalConvEncoder2d(
            in_channels,
            hidden_channels,
            hidden_channels,
            conv_params,
            act_norm=True,
            debug_show_dim=debug_show_dim,
        )
        self.bottleneck = PixelWiseConv2d(
            hidden_channels,
            latent_dim,
            act_norm=False,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        y, cs = self.cnn(x)
        z = self.bottleneck(y)
        return z, cs


class HierarchicalEncoder3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()

        self.cnn = HierarchicalConvEncoder3d(
            in_channels,
            hidden_channels,
            hidden_channels,
            conv_params,
            act_norm=True,
            debug_show_dim=debug_show_dim,
        )
        self.bottleneck = PixelWiseConv3d(
            hidden_channels,
            latent_dim,
            act_norm=False,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        y, cs = self.cnn(x)
        z = self.bottleneck(y)
        return z, cs


class HierarchicalDecoder2d(nn.Module):
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        aggregation: str,
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()
        self.bottleneck = PixelWiseConv2d(
            latent_dim,
            hidden_channels,
            act_norm=True,
        )
        self.cnn = HierarchicalConvDecoder2d(
            hidden_channels,
            out_channels,
            hidden_channels,
            conv_params,
            aggregation,
            debug_show_dim,
        )
        self.debug_show_dim = debug_show_dim

    def forward(self, z: Tensor, cs: list[Tensor]) -> Tensor:
        y = self.bottleneck(z)
        x = self.cnn(y, cs)

        if self.debug_show_dim:
            print("Latent", z.size())
            print("Output", y.size())
            print("Input", x.size())

        return x


class HierarchicalDecoder3d(nn.Module):
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        aggregation: str,
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()
        self.bottleneck = PixelWiseConv3d(
            latent_dim,
            hidden_channels,
            act_norm=True,
        )
        self.cnn = HierarchicalConvDecoder3d(
            hidden_channels,
            out_channels,
            hidden_channels,
            conv_params,
            aggregation,
            debug_show_dim,
        )
        self.debug_show_dim = debug_show_dim

    def forward(self, z: Tensor, cs: list[Tensor]) -> Tensor:
        y = self.bottleneck(z)
        x = self.cnn(y, cs)

        if self.debug_show_dim:
            print("Latent", z.size())
            print("Output", y.size())
            print("Input", x.size())

        return x


class HRDAE2d(nn.Module):
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
        connection_aggregation: str,
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()

        dec_hidden_channels = hidden_channels
        if aggregator == "concatenation":
            dec_hidden_channels += latent_dim
        self.content_encoder = HierarchicalEncoder2d(
            in_channels,
            hidden_channels,
            latent_dim,
            conv_params + [IdenticalConvBlockConvParams],
            debug_show_dim,
        )
        self.motion_encoder = motion_encoder
        self.decoder = HierarchicalDecoder2d(
            out_channels,
            dec_hidden_channels,
            2 * latent_dim if aggregator == "concatenation" else latent_dim,
            conv_params[::-1],
            connection_aggregation,
            debug_show_dim,
        )
        self.aggregator = create_aggregator2d(aggregator, latent_dim, latent_dim)
        # motion guided connection
        # (Mutual Suppression Network for Video Prediction using Disentangled Features)
        self.mgc = nn.ModuleList()
        for _ in conv_params:
            self.mgc.append(
                nn.Sequential(
                    create_aggregator2d(aggregator, hidden_channels, latent_dim),
                    ResNetBranch(
                        IdenticalConvBlock2d(dec_hidden_channels, dec_hidden_channels),
                        IdenticalConvBlock2d(
                            dec_hidden_channels, dec_hidden_channels, act_norm=False
                        ),
                    ),
                    nn.GroupNorm(2, dec_hidden_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
        self.activation = create_activation(activation)

    def forward(
        self,
        x_1d: Tensor,
        x_2d_0: Tensor,
        x_1d_0: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        c, cs = self.content_encoder(x_2d_0)
        m = self.motion_encoder(x_1d, x_1d_0)
        b, t, c_, h, w = m.size()
        m = m.reshape(b * t, c_, h, w)
        c_exp = c.repeat(t, 1, 1, 1)
        cs_exp = [c_.repeat(t, 1, 1, 1) for c_ in cs]

        assert len(self.mgc) == len(cs_exp)
        z = self.aggregator((c_exp, upsample_motion_tensor(m, c_exp)))
        for i, mgc in enumerate(self.mgc):
            cs_exp[i] = mgc((cs_exp[i], upsample_motion_tensor(m, cs_exp[i])))
        y = self.decoder(z, cs_exp[::-1])

        _, c_, h, w = y.size()
        y = y.reshape(b, t, c_, h, w)
        if self.activation is not None:
            y = self.activation(y)
        return y, [c] + cs, []


class CycleHRDAE2d(HRDAE2d):
    def forward(
        self,
        x_1d: Tensor,
        x_2d_0: Tensor,
        x_1d_0: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        y, cs, _ = super().forward(x_1d, x_2d_0, x_1d_0)
        b, t, c, h, w = y.size()
        y_seq = y.reshape(b * t, c, h, w)
        d, ds = self.content_encoder(y_seq)
        assert d.size(0) == b * t
        d = d.reshape(b, t, *d.size()[1:])
        assert len(cs) == 1 + len(ds)
        for i, di in enumerate(ds):
            assert di.size(0) == b * t
            ds[i] = di.reshape(b, t, *di.size()[1:])
        return y, [c.unsqueeze(1) for c in cs], [d] + ds


class HRDAE3d(nn.Module):
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
        connection_aggregation: str,
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()

        dec_hidden_channels = hidden_channels
        if aggregator == "concatenation":
            dec_hidden_channels += latent_dim
        self.content_encoder = HierarchicalEncoder3d(
            in_channels,
            hidden_channels,
            latent_dim,
            conv_params + [IdenticalConvBlockConvParams],
            debug_show_dim,
        )
        self.motion_encoder = motion_encoder
        self.decoder = HierarchicalDecoder3d(
            out_channels,
            dec_hidden_channels,
            2 * latent_dim if aggregator == "concatenation" else latent_dim,
            conv_params[::-1],
            connection_aggregation,
            debug_show_dim,
        )
        self.aggregator = create_aggregator2d(aggregator, latent_dim, latent_dim)
        # motion guided connection
        # (Mutual Suppression Network for Video Prediction using Disentangled Features)
        self.mgc = nn.ModuleList()
        for _ in conv_params:
            self.mgc.append(
                nn.Sequential(
                    create_aggregator3d(aggregator, hidden_channels, latent_dim),
                    ResNetBranch(
                        IdenticalConvBlock3d(dec_hidden_channels, dec_hidden_channels),
                        IdenticalConvBlock3d(
                            dec_hidden_channels, dec_hidden_channels, act_norm=False
                        ),
                    ),
                    nn.GroupNorm(2, dec_hidden_channels),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
        self.activation = create_activation(activation)

    def forward(
        self,
        x_2d: Tensor,
        x_3d_0: Tensor,
        x_2d_0: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        c, cs = self.content_encoder(x_3d_0)
        m = self.motion_encoder(x_2d, x_2d_0)
        b, t, c_, d, h, w = m.size()
        m = m.reshape(b * t, c_, d, h, w)
        c_exp = c.repeat(t, 1, 1, 1, 1)
        cs_exp = [c_.repeat(t, 1, 1, 1, 1) for c_ in cs]

        assert len(self.mgc) == len(cs_exp)
        z = self.aggregator((c_exp, upsample_motion_tensor(m, c_exp)))
        for i, mgc in enumerate(self.mgc):
            cs_exp[i] = mgc((cs_exp[i], upsample_motion_tensor(m, cs_exp[i])))
        y = self.decoder(z, cs_exp[::-1])

        _, c_, d, h, w = y.size()
        y = y.reshape(b, t, c_, d, h, w)
        if self.activation is not None:
            y = self.activation(y)
        return y, [c] + cs, []


class CycleHRDAE3d(HRDAE3d):
    def forward(
        self,
        x_2d: Tensor,
        x_3d_0: Tensor,
        x_2d_0: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        y, cs, _ = super().forward(x_2d, x_3d_0, x_2d_0)
        b, t, c, d_, h, w = y.size()
        y_seq = y.reshape(b * t, c, d_, h, w)
        d, ds = self.content_encoder(y_seq)

        assert d.size(0) == b * t
        d = d.reshape(b, t, *d.size()[1:])
        assert len(cs) == 1 + len(ds)
        for i, di in enumerate(ds):
            assert di.size(0) == b * t
            ds[i] = di.reshape(b, t, *di.size()[1:])
        return y, [c.unsqueeze(1) for c in cs], [d] + ds
