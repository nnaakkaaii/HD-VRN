# Licensed under the Apache License, Version 2.0 (the "License");
# Hierarchical Recurrent Disentangled 3D Video Reconstruction Network (HRD3D-VRN)

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field

from omegaconf import MISSING
from torch import Tensor, cat, nn
from torch.nn.functional import interpolate

from .modules import (ConvLSTM2d, ConvModule2d, GRU2d,
                      HierarchicalConvDecoder3d, HierarchicalConvEncoder3d,
                      IdenticalConvBlock3d, IdenticalConvBlockConvParams,
                      ResNetBranch, TCN2d)
from .option import NetworkOption


@dataclass
class HRD3DVRNOption(NetworkOption):
    content_encoder_in_channels: int = 2
    motion_encoder_in_channels: int = 1
    decoder_out_channels: int = 1
    latent_dim: int = 64
    content_encoder_conv_params: list[dict[str, int]] = field(
        default_factory=lambda: [{"kernel_size": 3, "stride": 2, "padding": 1}] * 3,
    )
    motion_encoder_conv_params: list[dict[str, int]] = field(
        default_factory=lambda: [{"kernel_size": 3, "stride": 2, "padding": 1}] * 3,
    )
    rnn2d: "MotionRNN2dOption" = MISSING
    debug_show_dim: bool = False


def create_hrd3dvrn(opt: HRD3DVRNOption) -> nn.Module:
    pass


class HierarchicalContentEncoder3d(HierarchicalConvEncoder3d):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, int]],
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            latent_dim,
            latent_dim,
            conv_params,
            debug_show_dim,
        )


class MotionRNN2d(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(
        self,
        x: Tensor,
        last_states: list[tuple[Tensor, Tensor]] | Tensor | None = None,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]] | Tensor | None]:
        pass


@dataclass
class MotionRNN2dOption(NetworkOption):
    pass


@dataclass
class MotionNoRNN2dOption(MotionRNN2dOption):
    pass


def create_motion_no_rnn2d(latent_dim: int, opt: MotionNoRNN2dOption) -> MotionRNN2d:
    return MotionNoRNN2d()


class MotionNoRNN2d(MotionRNN2d):
    def forward(
        self,
        x: Tensor,
        last_states: list[tuple[Tensor, Tensor]] | Tensor | None = None,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]] | Tensor | None]:
        return x, last_states


@dataclass
class MotionConvLSTM2dOption(MotionRNN2dOption):
    num_layers: int = 3


def create_motion_conv_lstm2d(
    latent_dim: int, opt: MotionConvLSTM2dOption
) -> MotionRNN2d:
    return MotionConvLSTM2d(latent_dim, opt.num_layers)


class MotionConvLSTM2d(MotionRNN2d):
    def __init__(
        self,
        latent_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.rnn = ConvLSTM2d(
            latent_dim,
            latent_dim,
            (3, 3),
            num_layers,
            batch_first=True,
        )

    def forward(
        self,
        x: Tensor,
        last_states: list[tuple[Tensor, Tensor]] | Tensor | None = None,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]] | Tensor | None]:
        assert last_states is None or (
            isinstance(last_states, list)
            and all([isinstance(s, tuple) for s in last_states])
        )
        y, last_states = self.rnn(x, last_states)
        assert isinstance(last_states, list) and all(
            [isinstance(s, tuple) for s in last_states]
        )
        return y, last_states


@dataclass
class MotionGRU2dOption(MotionRNN2dOption):
    num_layers: int = 3
    image_size: tuple[int, int] = (64, 64)


def create_motion_gru2d(latent_dim: int, opt: MotionGRU2dOption) -> MotionRNN2d:
    return MotionGRU2d(latent_dim, opt.num_layers, opt.image_size)


class MotionGRU2d(MotionRNN2d):
    def __init__(
        self,
        latent_dim: int,
        num_layers: int,
        image_size: tuple[int, int],
    ) -> None:
        super().__init__()
        self.rnn = GRU2d(
            latent_dim,
            latent_dim,
            num_layers,
            image_size,
        )

    def forward(
        self,
        x: Tensor,
        last_states: list[tuple[Tensor, Tensor]] | Tensor | None = None,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]] | Tensor | None]:
        assert last_states is None or isinstance(last_states, Tensor)
        y, last_states = self.rnn(x, last_states)
        assert isinstance(last_states, Tensor)
        return y, last_states


@dataclass
class MotionTCN2dOption(MotionRNN2dOption):
    num_layers: int = 3
    image_size: tuple[int, int] = (64, 64)
    kernel_size: int = 3
    dropout: float = 0.0


def create_motion_tcn2d(latent_dim: int, opt: MotionTCN2dOption) -> MotionRNN2d:
    return MotionTCN2d(
        latent_dim, opt.num_layers, opt.image_size, opt.kernel_size, opt.dropout
    )


class MotionTCN2d(MotionRNN2d):
    def __init__(
        self,
        latent_dim: int,
        num_layers: int,
        image_size: tuple[int, int],
        kernel_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rnn = TCN2d(
            latent_dim,
            [latent_dim] * num_layers,
            kernel_size,
            image_size,
            dropout,
        )

    def forward(
        self,
        x: Tensor,
        last_states: list[tuple[Tensor, Tensor]] | Tensor | None = None,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]] | Tensor | None]:
        assert last_states is None
        y = self.rnn(x)
        return y, None


class MotionEncoder2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, int]],
        rnn2d: MotionRNN2d,
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()

        self.enc = ConvModule2d(
            in_channels,
            latent_dim,
            latent_dim,
            conv_params,
            transpose=False,
            debug_show_dim=debug_show_dim,
        )
        self.rnn = rnn2d
        self.debug_show_dim = debug_show_dim

    def forward(self, x: Tensor) -> Tensor:
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)
        x = self.enc(x)
        _, c, h, w = x.size()
        x = x.view(b, t, c, h, w)
        x, _ = self.rnn(x)
        if self.debug_show_dim:
            print(f"{self.__class__.__name__}", x.size())
        return x


class HierarchicalDecoder3d(nn.Module):
    def __init__(
        self,
        out_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, int]],
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()

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

    def forward(self, m: Tensor, c: Tensor, cs: list[Tensor]) -> Tensor:
        assert len(self.mgc) == len(cs)

        x = cat([c, self._upsample_motion_tensor(m, c)], dim=1)
        for i, (mgc, c_) in enumerate(zip(self.mgc, cs)):
            c_ = cat([c_, self._upsample_motion_tensor(m, c_)], dim=1)
            cs[i] = mgc(c_)

        return self.dec(x, cs)

    @staticmethod
    def _upsample_motion_tensor(m: Tensor, c: Tensor) -> Tensor:
        b, c_, d, h, w = c.size()
        m = m.unsqueeze(2)
        m = interpolate(m, size=(d, h, w), mode="trilinear", align_corners=True)
        return m


class HRD3DVRN(nn.Module):
    def __init__(
        self,
        content_encoder_in_channels: int,
        motion_encoder_in_channels: int,
        decoder_out_channels: int,
        latent_dim: int,
        content_encoder_conv_params: list[dict[str, int]],
        motion_encoder_conv_params: list[dict[str, int]],
        rnn2d: MotionRNN2d,
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()
        self.content_encoder = HierarchicalContentEncoder3d(
            content_encoder_in_channels,
            latent_dim,
            content_encoder_conv_params + [IdenticalConvBlockConvParams],
            debug_show_dim,
        )
        self.motion_encoder = MotionEncoder2d(
            motion_encoder_in_channels,
            latent_dim,
            motion_encoder_conv_params,
            rnn2d,
            debug_show_dim,
        )
        self.decoder = HierarchicalDecoder3d(
            decoder_out_channels,
            2 * latent_dim,
            content_encoder_conv_params[::-1],
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
                {"kernel_size": 3, "stride": 2, "padding": 1},
                {"kernel_size": 3, "stride": 2, "padding": 1},
                {"kernel_size": 3, "stride": 1, "padding": 1},
            ],
            debug_show_dim=True,
        )
        x = randn(8, 1, 64, 64, 64)
        c, cs = ce_net(x)
        print("c input", x.size())
        for i, c_ in enumerate(cs):
            print(f"c hidden{i}", c_.size())
        print("c output", c.size())

        me_net = MotionEncoder2d(
            1,
            16,
            [
                {"kernel_size": 3, "stride": 2, "padding": 1},
                {"kernel_size": 3, "stride": 2, "padding": 1},
                {"kernel_size": 3, "stride": 1, "padding": 1},
            ],
            MotionConvLSTM2d(16, 2),
            debug_show_dim=True,
        )
        x = randn(8, 10, 1, 64, 64)
        m = me_net(x)
        print("m input", x.size())
        print("m output", m.size())

        d_net = HierarchicalDecoder3d(
            1,
            2 * 16,
            [
                {"kernel_size": 3, "stride": 2, "padding": 1},
                {"kernel_size": 3, "stride": 2, "padding": 1},
            ],
            debug_show_dim=True,
        )
        d = d_net(
            m.view(8 * 10, 16, 16, 16),
            c.repeat(10, 1, 1, 1, 1),
            [c_.repeat(10, 1, 1, 1, 1) for c_ in cs[::-1]],
        )
        print("d output", d.size())

    def test2():
        from torch import randn

        net = HRD3DVRN(
            content_encoder_in_channels=2,
            motion_encoder_in_channels=1,
            decoder_out_channels=1,
            latent_dim=16,
            content_encoder_conv_params=[
                {"kernel_size": 3, "stride": 2, "padding": 1},
                {"kernel_size": 3, "stride": 2, "padding": 1},
            ],
            motion_encoder_conv_params=[
                {"kernel_size": 3, "stride": 2, "padding": 1},
                {"kernel_size": 3, "stride": 2, "padding": 1},
                {"kernel_size": 3, "stride": 2, "padding": 1},
            ],
            rnn2d=MotionConvLSTM2d(16, 2),
            debug_show_dim=True,
        )
        x = net(
            randn(8, 10, 1, 64, 64),
            randn(8, 2, 64, 64, 64),
        )
        print(x.size())

    test2()
