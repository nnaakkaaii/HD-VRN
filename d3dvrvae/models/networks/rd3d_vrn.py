# Licensed under the Apache License, Version 2.0 (the "License");
# Recurrent Disentangled 3D Video Reconstruction Network (HRD3D-VRN)

from dataclasses import dataclass, field

from omegaconf import MISSING
from torch import Tensor, cat, nn
from torch.nn.functional import interpolate

from .modules import ConvModule3d, IdenticalConvBlockConvParams
from .motion_encoder import (MotionEncoder2d, MotionEncoder2dOption,
                             create_motion_encoder2d)
from .option import NetworkOption


@dataclass
class RD3DVRNOption(NetworkOption):
    in_channels: int = 2
    out_channels: int = 1
    latent_dim: int = 64
    conv_params: list[dict[str, list[int]]] = field(
        default_factory=lambda: [{"kernel_size": [3], "stride": [2], "padding": [1]}] * 3,
    )
    motion_encoder2d: MotionEncoder2dOption = MISSING
    debug_show_dim: bool = False


def create_rd3dvrn(opt: RD3DVRNOption) -> nn.Module:
    motion_encoder2d = create_motion_encoder2d(
        opt.latent_dim, opt.debug_show_dim, opt.motion_encoder2d
    )
    return RD3DVRN(
        opt.in_channels,
        opt.out_channels,
        opt.latent_dim,
        opt.conv_params,
        motion_encoder2d,
        opt.debug_show_dim,
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


class NormalDecoder3d(nn.Module):
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

    def forward(self, m: Tensor, c: Tensor) -> Tensor:
        x = cat([c, self._upsample_motion_tensor(m, c)], dim=1)
        return self.dec(x)

    @staticmethod
    def _upsample_motion_tensor(m: Tensor, c: Tensor) -> Tensor:
        b, c_, d, h, w = c.size()
        m = m.unsqueeze(2)
        m = interpolate(m, size=(d, h, w), mode="trilinear", align_corners=True)
        return m


class RD3DVRN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        motion_encoder2d: MotionEncoder2d,
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()
        self.content_encoder = NormalContentEncoder3d(
            in_channels,
            latent_dim,
            conv_params + [IdenticalConvBlockConvParams],
            debug_show_dim,
        )
        self.motion_encoder = motion_encoder2d
        self.decoder = NormalDecoder3d(
            out_channels,
            2 * latent_dim,
            conv_params[::-1],
            debug_show_dim,
        )

    def forward(
        self,
        x_2d: Tensor,
        x_3d_0: Tensor,
    ) -> Tensor:
        c = self.content_encoder(x_3d_0)
        m = self.motion_encoder(x_2d)
        b, t, c_, h, w = m.size()
        m = m.view(b * t, c_, h, w)
        c = c.repeat(t, 1, 1, 1, 1)
        return self.decoder(m, c)


if __name__ == "__main__":

    def test():
        from torch import randn

        from .motion_encoder import MotionRNNEncoder2dOption
        from .rnn import ConvLSTM2dOption

        option = RD3DVRNOption(
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
            debug_show_dim=True,
        )
        net = create_rd3dvrn(option)
        x = net(
            randn(8, 10, 1, 64, 64),
            randn(8, 2, 64, 64, 64),
        )
        print(x.size())

    test()
