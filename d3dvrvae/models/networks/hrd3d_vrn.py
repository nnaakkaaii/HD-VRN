# Licensed under the Apache License, Version 2.0 (the "License");
# Hierarchical Recurrent Disentangled 3D Video Reconstruction Network (HRD3D-VRN)

from dataclasses import dataclass

from torch import Tensor, nn

from .modules import (HierarchicalConvDecoder3d, HierarchicalConvEncoder3d,
                      IdenticalConvBlock3d, IdenticalConvBlockConvParams,
                      ResNetBranch)
from .motion_encoder import MotionEncoder2d, create_motion_encoder2d
from .rd3d_vrn import RD3DVRNOption
from .functions import aggregate


@dataclass
class HRD3DVRNOption(RD3DVRNOption):
    pass


def create_hrd3dvrn(opt: HRD3DVRNOption) -> nn.Module:
    motion_encoder2d = create_motion_encoder2d(
        opt.latent_dim, opt.debug_show_dim, opt.motion_encoder2d
    )
    return HRD3DVRN(
        opt.in_channels,
        opt.out_channels,
        opt.latent_dim,
        opt.conv_params,
        motion_encoder2d,
        opt.aggregation_method,
        opt.debug_show_dim,
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
        assert aggregation_method in ["concat", "sum"], f"aggregation_method: {aggregation_method} not implemented"

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


class HRD3DVRN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        motion_encoder2d: MotionEncoder2d,
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
        self.motion_encoder = motion_encoder2d
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
    ) -> Tensor:
        c, cs = self.content_encoder(x_3d_0)
        m = self.motion_encoder(x_2d)
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
            [c_.repeat(10, 1, 1, 1, 1) for c_ in cs[::-1]],  # [(80, 16, 16, 16, 16), (80, 16, 32, 32, 32)]
        )
        print("d output", d.size())

    def test2():
        from torch import randn

        from .motion_encoder import MotionRNNEncoder2dOption
        from .rnn import ConvLSTM2dOption

        option = HRD3DVRNOption(
            in_channels=2,
            out_channels=1,
            latent_dim=16,
            conv_params=[
                {"kernel_size": [3], "stride": [2], "padding": [1]},
                {"kernel_size": [3], "stride": [2], "padding": [1]},
            ],
            motion_encoder2d=MotionRNNEncoder2dOption(
                in_channels=1,
                conv_params=[
                    {"kernel_size": [3], "stride": [2], "padding": [1]},
                    {"kernel_size": [3], "stride": [2], "padding": [1]},
                    {"kernel_size": [3], "stride": [2], "padding": [1]},
                ],
                rnn2d=ConvLSTM2dOption(
                    num_layers=1,
                ),
            ),
            aggregation_method="concat",
            debug_show_dim=True,
        )
        net = create_hrd3dvrn(option)
        x = net(
            randn(8, 10, 1, 64, 64),
            randn(8, 2, 64, 64, 64),
        )
        print(x.size())

    test1()
    test2()
