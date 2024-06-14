# Licensed under the Apache License, Version 2.0 (the "License");
# Recurrent AutoEncoder

from dataclasses import dataclass, field

from omegaconf import MISSING
from torch import Tensor, nn
from torch.nn.functional import interpolate

from .modules import ConvModule2d, ConvModule3d
from .motion_encoder import (MotionEncoder1d, MotionEncoder1dOption,
                             MotionEncoder2d, MotionEncoder2dOption,
                             create_motion_encoder1d, create_motion_encoder2d)
from .option import NetworkOption


@dataclass
class RAE2dOption(NetworkOption):
    out_channels: int = 1
    latent_dim: int = 64
    conv_params: list[dict[str, list[int]]] = field(
        default_factory=lambda: [{"kernel_size": [3], "stride": [2], "padding": [1]}]
        * 3,
    )
    motion_encoder1d: MotionEncoder1dOption = MISSING
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
    motion_encoder2d: MotionEncoder2dOption = MISSING
    upsample_size: list[int] = field(default_factory=lambda: [8, 8, 8])
    debug_show_dim: bool = False


def create_rae2d(opt: RAE2dOption) -> nn.Module:
    motion_encoder1d = create_motion_encoder1d(
        opt.latent_dim, opt.debug_show_dim, opt.motion_encoder1d
    )
    return RAE2d(
        opt.out_channels,
        opt.latent_dim,
        opt.conv_params,
        motion_encoder1d,
        opt.upsample_size,
        opt.debug_show_dim,
    )


def create_rae3d(opt: RAE3dOption) -> nn.Module:
    motion_encoder2d = create_motion_encoder2d(
        opt.latent_dim, opt.debug_show_dim, opt.motion_encoder2d
    )
    return RAE3d(
        opt.out_channels,
        opt.latent_dim,
        opt.conv_params,
        motion_encoder2d,
        opt.upsample_size,
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
        motion_encoder1d: MotionEncoder1d,
        upsample_size: list[int],
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()
        self.motion_encoder = motion_encoder1d
        self.decoder = Decoder2d(
            out_channels,
            latent_dim,
            conv_params[::-1],
            debug_show_dim,
        )
        self.upsample_size = upsample_size

    def forward(
        self,
        x_1d: Tensor,
    ) -> Tensor:
        m = self.motion_encoder(x_1d)
        b, t, c_, h = m.size()
        m = m.view(b * t, c_, h, 1)
        m = interpolate(m, size=self.upsample_size, mode="bilinear", align_corners=True)
        return self.decoder(m)


class RAE3d(nn.Module):
    def __init__(
        self,
        out_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        motion_encoder2d: MotionEncoder2d,
        upsample_size: list[int],
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()
        self.motion_encoder = motion_encoder2d
        self.decoder = Decoder3d(
            out_channels,
            latent_dim,
            conv_params[::-1],
            debug_show_dim,
        )
        self.upsample_size = upsample_size

    def forward(
        self,
        x_2d: Tensor,
    ) -> Tensor:
        m = self.motion_encoder(x_2d)
        b, t, c_, d, h = m.size()
        m = m.view(b * t, c_, d, h, 1)
        m = interpolate(
            m, size=self.upsample_size, mode="trilinear", align_corners=True
        )
        return self.decoder(m)


if __name__ == "__main__":

    def test():
        from torch import randn

        from .motion_encoder import (MotionRNNEncoder1dOption,
                                     MotionRNNEncoder2dOption)
        from .rnn import ConvLSTM1dOption, ConvLSTM2dOption

        option = RAE2dOption(
            out_channels=1,
            latent_dim=16,
            conv_params=[
                {"kernel_size": [3], "stride": [2], "padding": [1]},
                {"kernel_size": [3], "stride": [2], "padding": [1]},
                {"kernel_size": [3], "stride": [2], "padding": [1]},
            ],
            motion_encoder1d=MotionRNNEncoder1dOption(
                in_channels=1,
                conv_params=[
                    {"kernel_size": [3], "stride": [2], "padding": [1]},
                    {"kernel_size": [3], "stride": [2], "padding": [1]},
                    {"kernel_size": [3], "stride": [2], "padding": [1]},
                ],
                rnn1d=ConvLSTM1dOption(
                    num_layers=1,
                ),
            ),
            debug_show_dim=True,
        )
        net = create_rae2d(option)
        x = net(
            randn(8, 10, 1, 64),
        )
        print(x.size())

        option = RAE3dOption(
            out_channels=1,
            latent_dim=16,
            conv_params=[
                {"kernel_size": [3], "stride": [2], "padding": [1]},
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
        net = create_rae3d(option)
        x = net(
            randn(8, 10, 1, 64, 64),
        )
        print(x.size())

    test()
