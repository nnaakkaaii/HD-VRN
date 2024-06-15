# Licensed under the Apache License, Version 2.0 (the "License");
# Hierarchical Recurrent Disentangled AutoEncoder (HR-DAE)

from dataclasses import dataclass

from torch import Tensor, nn

from .functions import aggregate
from .modules import (
    HierarchicalConvDecoder2d,
    HierarchicalConvDecoder3d,
    HierarchicalConvEncoder2d,
    HierarchicalConvEncoder3d,
    IdenticalConvBlock2d,
    IdenticalConvBlock3d,
    IdenticalConvBlockConvParams,
    ResNetBranch,
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
    pass


@dataclass
class HRDAE3dOption(RDAE3dOption):
    pass


def create_hrdae2d(opt: HRDAE2dOption) -> nn.Module:
    motion_encoder = create_motion_encoder1d(
        opt.latent_dim, opt.debug_show_dim, opt.motion_encoder
    )
    return HRDAE2d(
        opt.in_channels,
        opt.out_channels,
        opt.latent_dim,
        opt.conv_params,
        motion_encoder,
        opt.aggregation_method,
        opt.debug_show_dim,
    )


def create_hrdae3d(opt: HRDAE3dOption) -> nn.Module:
    motion_encoder = create_motion_encoder2d(
        opt.latent_dim, opt.debug_show_dim, opt.motion_encoder
    )
    return HRDAE3d(
        opt.in_channels,
        opt.out_channels,
        opt.latent_dim,
        opt.conv_params,
        motion_encoder,
        opt.aggregation_method,
        opt.debug_show_dim,
    )


class HierarchicalContentEncoder2d(HierarchicalConvEncoder2d):
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
            debug_show_dim,
        )


class HierarchicalContentEncoder3d(HierarchicalConvEncoder3d):
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
            debug_show_dim,
        )


class HierarchicalDecoder2d(nn.Module):
    def __init__(
        self,
        out_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        aggregation_method: str = "concat",
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()
        assert aggregation_method in [
            "concat",
            "sum",
        ], f"aggregation_method: {aggregation_method} not implemented"

        if aggregation_method == "concat":
            latent_dim *= 2

        self.dec = HierarchicalConvDecoder2d(
            latent_dim,
            out_channels,
            latent_dim,
            conv_params,
            debug_show_dim,
        )
        # motion guided connection
        # (Mutual Suppression Network for Video Prediction using Disentangled Features)
        self.mgc = nn.ModuleList()
        for _ in conv_params:
            self.mgc.append(
                nn.Sequential(
                    ResNetBranch(
                        IdenticalConvBlock2d(latent_dim, latent_dim),
                        IdenticalConvBlock2d(latent_dim, latent_dim, act_norm=False),
                    ),
                    nn.GroupNorm(2, latent_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
        self.aggregation_method = aggregation_method

    def forward(self, m: Tensor, c: Tensor, cs: list[Tensor]) -> Tensor:
        assert len(self.mgc) == len(cs)

        x = aggregate(m, c, method=self.aggregation_method)
        for i, (mgc, c_) in enumerate(zip(self.mgc, cs)):
            c_ = aggregate(m, c_, method=self.aggregation_method)
            cs[i] = mgc(c_)

        return self.dec(x, cs)


class HierarchicalDecoder3d(nn.Module):
    def __init__(
        self,
        out_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        aggregation_method: str = "concat",
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()
        assert aggregation_method in [
            "concat",
            "sum",
        ], f"aggregation_method: {aggregation_method} not implemented"

        if aggregation_method == "concat":
            latent_dim *= 2

        self.dec = HierarchicalConvDecoder3d(
            latent_dim,
            out_channels,
            latent_dim,
            conv_params,
            debug_show_dim,
        )
        # motion guided connection
        # (Mutual Suppression Network for Video Prediction using Disentangled Features)
        self.mgc = nn.ModuleList()
        for _ in conv_params:
            self.mgc.append(
                nn.Sequential(
                    ResNetBranch(
                        IdenticalConvBlock3d(latent_dim, latent_dim),
                        IdenticalConvBlock3d(latent_dim, latent_dim, act_norm=False),
                    ),
                    nn.GroupNorm(2, latent_dim),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
        self.aggregation_method = aggregation_method

    def forward(self, m: Tensor, c: Tensor, cs: list[Tensor]) -> Tensor:
        assert len(self.mgc) == len(cs)

        x = aggregate(m, c, method=self.aggregation_method)
        for i, (mgc, c_) in enumerate(zip(self.mgc, cs)):
            c_ = aggregate(m, c_, method=self.aggregation_method)
            cs[i] = mgc(c_)

        return self.dec(x, cs)


class HRDAE2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        motion_encoder: MotionEncoder1d,
        aggregation_method: str = "concat",
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()
        self.content_encoder = HierarchicalContentEncoder2d(
            in_channels,
            latent_dim,
            conv_params + [IdenticalConvBlockConvParams],
            debug_show_dim,
        )
        self.motion_encoder = motion_encoder
        self.decoder = HierarchicalDecoder2d(
            out_channels,
            latent_dim,
            conv_params[::-1],
            aggregation_method,
            debug_show_dim,
        )

    def forward(
        self,
        x_1d: Tensor,
        x_2d_0: Tensor,
        x_1d_0: Tensor | None = None,
    ) -> Tensor:
        c, cs = self.content_encoder(x_2d_0)
        m = self.motion_encoder(x_1d, x_1d_0)
        b, t, c_, h = m.size()
        m = m.view(b * t, c_, h)
        c = c.repeat(t, 1, 1, 1)
        cs = [c_.repeat(t, 1, 1, 1) for c_ in cs]
        y = self.decoder(m, c, cs[::-1])
        _, c_, h, w = y.size()
        return y.view(b, t, c_, h, w)


class HRDAE3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        motion_encoder: MotionEncoder2d,
        aggregation_method: str = "concat",
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()
        self.content_encoder = HierarchicalContentEncoder3d(
            in_channels,
            latent_dim,
            conv_params + [IdenticalConvBlockConvParams],
            debug_show_dim,
        )
        self.motion_encoder = motion_encoder
        self.decoder = HierarchicalDecoder3d(
            out_channels,
            latent_dim,
            conv_params[::-1],
            aggregation_method,
            debug_show_dim,
        )

    def forward(
        self,
        x_2d: Tensor,
        x_3d_0: Tensor,
        x_2d_0: Tensor | None = None,
    ) -> Tensor:
        c, cs = self.content_encoder(x_3d_0)
        m = self.motion_encoder(x_2d, x_2d_0)
        b, t, c_, h, w = m.size()
        m = m.view(b * t, c_, h, w)
        c = c.repeat(t, 1, 1, 1, 1)
        cs = [c_.repeat(t, 1, 1, 1, 1) for c_ in cs]
        y = self.decoder(m, c, cs[::-1])
        _, c_, d, h, w = y.size()
        return y.view(b, t, c_, d, h, w)
