# Licensed under the Apache License, Version 2.0 (the "License");
# Hierarchical Recurrent Disentangled AutoEncoder (HR-DAE)

from dataclasses import dataclass

from torch import Tensor, nn

from .functions import aggregate
from .modules import (HierarchicalConvDecoder2d, HierarchicalConvDecoder3d,
                      HierarchicalConvEncoder2d, HierarchicalConvEncoder3d,
                      IdenticalConvBlock2d, IdenticalConvBlock3d,
                      IdenticalConvBlockConvParams, ResNetBranch)
from .motion_encoder import (MotionEncoder1d, MotionEncoder2d,
                             create_motion_encoder1d, create_motion_encoder2d)
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
        x_1d_0: Tensor,
    ) -> Tensor:
        c, cs = self.content_encoder(x_2d_0)
        m = self.motion_encoder(x_1d, x_1d_0)
        b, t, c_, h = m.size()
        m = m.view(b * t, c_, h)
        c = c.repeat(t, 1, 1, 1)
        cs = [c_.repeat(t, 1, 1, 1) for c_ in cs]
        return self.decoder(m, c, cs[::-1])


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
        x_2d_0: Tensor,
    ) -> Tensor:
        c, cs = self.content_encoder(x_3d_0)
        m = self.motion_encoder(x_2d, x_2d_0)
        b, t, c_, h, w = m.size()
        m = m.view(b * t, c_, h, w)
        c = c.repeat(t, 1, 1, 1, 1)
        cs = [c_.repeat(t, 1, 1, 1, 1) for c_ in cs]
        return self.decoder(m, c, cs[::-1])


if __name__ == "__main__":

    def test1():
        from torch import randn

        ce_net = HierarchicalContentEncoder3d(
            1,
            16,
            [
                {"kernel_size": [3], "stride": [2], "padding": [1]},
                {"kernel_size": [3], "stride": [2], "padding": [1]},
                {"kernel_size": [3], "stride": [1], "padding": [1]},
            ],
            debug_show_dim=True,
        )
        x = randn(8, 1, 64, 64, 64)
        c, cs = ce_net(x)
        print("c input", x.size())
        for i, c_ in enumerate(cs):
            print(f"c hidden{i}", c_.size())
        print("c output", c.size())

        d_net = HierarchicalDecoder3d(
            1,
            16,
            [
                {"kernel_size": [3], "stride": [2], "padding": [1]},
                {"kernel_size": [3], "stride": [2], "padding": [1]},
            ],
            aggregation_method="concat",
            debug_show_dim=True,
        )
        d = d_net(
            randn(8 * 10, 16, 8, 8),
            c.repeat(10, 1, 1, 1, 1),  # (80, 16, 16, 16, 16)
            [
                c_.repeat(10, 1, 1, 1, 1) for c_ in cs[::-1]
            ],  # [(80, 16, 16, 16, 16), (80, 16, 32, 32, 32)]
        )
        print("d output", d.size())

    def test2():
        from torch import randn

        from .motion_encoder import (MotionRNNEncoder1dOption,
                                     MotionRNNEncoder2dOption)
        from .rnn import ConvLSTM1dOption, ConvLSTM2dOption

        option = HRDAE2dOption(
            in_channels=2,
            out_channels=1,
            latent_dim=16,
            conv_params=[
                {"kernel_size": [3], "stride": [2], "padding": [1]},
                {"kernel_size": [3], "stride": [2], "padding": [1]},
            ],
            motion_encoder=MotionRNNEncoder1dOption(
                in_channels=1,
                conv_params=[
                    {"kernel_size": [3], "stride": [2], "padding": [1]},
                    {"kernel_size": [3], "stride": [2], "padding": [1]},
                    {"kernel_size": [3], "stride": [2], "padding": [1]},
                ],
                rnn=ConvLSTM1dOption(
                    num_layers=1,
                ),
            ),
            aggregation_method="concat",
            debug_show_dim=True,
        )
        net = create_hrdae2d(option)
        x = net(
            randn(8, 10, 1, 64),
            randn(8, 2, 64, 64),
        )
        print(x.size())

        option = HRDAE3dOption(
            in_channels=2,
            out_channels=1,
            latent_dim=16,
            conv_params=[
                {"kernel_size": [3], "stride": [2], "padding": [1]},
                {"kernel_size": [3], "stride": [2], "padding": [1]},
            ],
            motion_encoder=MotionRNNEncoder2dOption(
                in_channels=1,
                conv_params=[
                    {"kernel_size": [3], "stride": [2], "padding": [1]},
                    {"kernel_size": [3], "stride": [2], "padding": [1]},
                    {"kernel_size": [3], "stride": [2], "padding": [1]},
                ],
                rnn=ConvLSTM2dOption(
                    num_layers=1,
                ),
            ),
            aggregation_method="concat",
            debug_show_dim=True,
        )
        net = create_hrdae3d(option)
        x = net(
            randn(8, 10, 1, 64, 64),
            randn(8, 2, 64, 64, 64),
        )
        print(x.size())

    # test1()
    test2()
