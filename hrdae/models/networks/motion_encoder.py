from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field

from omegaconf import MISSING
from torch import Tensor, nn

from .modules import ConvModule1d, ConvModule2d, ConvModule3d
from .rnn import (RNN1d, RNN1dOption, RNN2d, RNN2dOption, create_rnn1d,
                  create_rnn2d)


@dataclass
class MotionEncoder1dOption:
    pass


@dataclass
class MotionEncoder2dOption:
    pass


def create_motion_encoder1d(
    latent_dim: int, debug_show_dim: bool, opt: MotionEncoder1dOption
) -> "MotionEncoder1d":
    if (
        isinstance(opt, MotionRNNEncoder1dOption)
        and type(opt) is MotionRNNEncoder1dOption
    ):
        return create_motion_rnn_encoder1d(latent_dim, debug_show_dim, opt)
    if (
        isinstance(opt, MotionNormalEncoder1dOption)
        and type(opt) is MotionNormalEncoder1dOption
    ):
        return create_motion_normal_encoder1d(latent_dim, debug_show_dim, opt)
    if (
        isinstance(opt, MotionConv2dEncoder1dOption)
        and type(opt) is MotionConv2dEncoder1dOption
    ):
        return create_motion_conv2d_encoder1d(latent_dim, debug_show_dim, opt)
    if (
        isinstance(opt, MotionGuidedEncoder1dOption)
        and type(opt) is MotionGuidedEncoder1dOption
    ):
        return create_motion_guided_encoder1d(latent_dim, debug_show_dim, opt)
    if (
        isinstance(opt, MotionTSNEncoder1dOption)
        and type(opt) is MotionTSNEncoder1dOption
    ):
        return create_motion_tsn_encoder1d(latent_dim, debug_show_dim, opt)
    raise NotImplementedError(f"{opt.__class__.__name__} not implemented")


def create_motion_encoder2d(
    latent_dim: int, debug_show_dim: bool, opt: MotionEncoder2dOption
) -> "MotionEncoder2d":
    if (
        isinstance(opt, MotionRNNEncoder2dOption)
        and type(opt) is MotionRNNEncoder2dOption
    ):
        return create_motion_rnn_encoder2d(latent_dim, debug_show_dim, opt)
    if (
        isinstance(opt, MotionNormalEncoder2dOption)
        and type(opt) is MotionNormalEncoder2dOption
    ):
        return create_motion_normal_encoder2d(latent_dim, debug_show_dim, opt)
    if (
        isinstance(opt, MotionConv3dEncoder2dOption)
        and type(opt) is MotionConv3dEncoder2dOption
    ):
        return create_motion_conv3d_encoder2d(latent_dim, debug_show_dim, opt)
    if (
        isinstance(opt, MotionGuidedEncoder2dOption)
        and type(opt) is MotionGuidedEncoder2dOption
    ):
        return create_motion_guided_encoder2d(latent_dim, debug_show_dim, opt)
    if (
        isinstance(opt, MotionTSNEncoder2dOption)
        and type(opt) is MotionTSNEncoder2dOption
    ):
        return create_motion_tsn_encoder2d(latent_dim, debug_show_dim, opt)
    raise NotImplementedError(f"{opt.__class__.__name__} not implemented")


class MotionEncoder1d(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(
        self,
        x: Tensor,
        x_0: Tensor | None = None,
    ) -> Tensor:
        pass


class MotionEncoder2d(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(
        self,
        x: Tensor,
        x_0: Tensor | None = None,
    ) -> Tensor:
        pass


@dataclass
class MotionNormalEncoder1dOption(MotionEncoder1dOption):
    in_channels: int
    conv_params: list[dict[str, list[int]]] = field(
        default_factory=lambda: [{"kernel_size": [3], "stride": [2], "padding": [1]}]
        * 3,
    )


@dataclass
class MotionNormalEncoder2dOption(MotionEncoder2dOption):
    in_channels: int
    conv_params: list[dict[str, list[int]]] = field(
        default_factory=lambda: [{"kernel_size": [3], "stride": [2], "padding": [1]}]
        * 3,
    )


def create_motion_normal_encoder1d(
    latent_dim: int, debug_show_dim: bool, opt: MotionNormalEncoder1dOption
) -> MotionEncoder1d:
    return MotionNormalEncoder1d(
        opt.in_channels,
        latent_dim,
        opt.conv_params,
        debug_show_dim=debug_show_dim,
    )


def create_motion_normal_encoder2d(
    latent_dim: int, debug_show_dim: bool, opt: MotionNormalEncoder2dOption
) -> MotionEncoder2d:
    return MotionNormalEncoder2d(
        opt.in_channels,
        latent_dim,
        opt.conv_params,
        debug_show_dim=debug_show_dim,
    )


class MotionNormalEncoder1d(MotionEncoder1d):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()

        self.enc = ConvModule1d(
            in_channels,
            latent_dim,
            latent_dim,
            conv_params,
            transpose=False,
            debug_show_dim=debug_show_dim,
        )
        self.debug_show_dim = debug_show_dim

    def forward(
        self,
        x: Tensor,
        x_0: Tensor | None = None,
    ) -> Tensor:
        b, t, c, h = x.size()
        x = x.view(b * t, c, h)
        x = self.enc(x)
        _, c, h = x.size()
        x = x.view(b, t, c, h)
        if self.debug_show_dim:
            print(f"{self.__class__.__name__}", x.size())
        return x


class MotionNormalEncoder2d(MotionEncoder2d):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
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
        self.debug_show_dim = debug_show_dim

    def forward(
        self,
        x: Tensor,
        x_0: Tensor | None = None,
    ) -> Tensor:
        b, t, c, d, h = x.size()
        x = x.view(b * t, c, d, h)
        x = self.enc(x)
        _, c, d, h = x.size()
        x = x.view(b, t, c, d, h)
        if self.debug_show_dim:
            print(f"{self.__class__.__name__}", x.size())
        return x


@dataclass
class MotionRNNEncoder1dOption(MotionNormalEncoder1dOption):
    rnn: RNN1dOption = MISSING


@dataclass
class MotionRNNEncoder2dOption(MotionNormalEncoder2dOption):
    rnn: RNN2dOption = MISSING


def create_motion_rnn_encoder1d(
    latent_dim: int, debug_show_dim: bool, opt: MotionRNNEncoder1dOption
) -> MotionEncoder1d:
    return MotionRNNEncoder1d(
        opt.in_channels,
        latent_dim,
        opt.conv_params,
        create_rnn1d(latent_dim, opt.rnn),
        debug_show_dim=debug_show_dim,
    )


def create_motion_rnn_encoder2d(
    latent_dim: int, debug_show_dim: bool, opt: MotionRNNEncoder2dOption
) -> MotionEncoder2d:
    return MotionRNNEncoder2d(
        opt.in_channels,
        latent_dim,
        opt.conv_params,
        create_rnn2d(latent_dim, opt.rnn),
        debug_show_dim=debug_show_dim,
    )


class MotionRNNEncoder1d(MotionNormalEncoder1d):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        rnn: RNN1d,
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            latent_dim,
            conv_params,
            debug_show_dim=debug_show_dim,
        )
        self.rnn = rnn

    def forward(
        self,
        x: Tensor,
        x_0: Tensor | None = None,
    ) -> Tensor:
        x = super().forward(x, x_0)
        x, _ = self.rnn(x)
        if self.debug_show_dim:
            print(f"{self.__class__.__name__}", x.size())
        return x


class MotionRNNEncoder2d(MotionNormalEncoder2d):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        rnn: RNN2d,
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            latent_dim,
            conv_params,
            debug_show_dim=debug_show_dim,
        )
        self.rnn = rnn

    def forward(
        self,
        x: Tensor,
        x_0: Tensor | None = None,
    ) -> Tensor:
        x = super().forward(x, x_0)
        x, _ = self.rnn(x)
        if self.debug_show_dim:
            print(f"{self.__class__.__name__}", x.size())
        return x


@dataclass
class MotionConv2dEncoder1dOption(MotionNormalEncoder1dOption):
    pass


@dataclass
class MotionConv3dEncoder2dOption(MotionNormalEncoder2dOption):
    pass


def create_motion_conv2d_encoder1d(
    latent_dim: int, debug_show_dim: bool, opt: MotionConv2dEncoder1dOption
) -> MotionEncoder1d:
    return MotionConv2dEncoder1d(
        opt.in_channels,
        latent_dim,
        opt.conv_params,
        debug_show_dim=debug_show_dim,
    )


def create_motion_conv3d_encoder2d(
    latent_dim: int, debug_show_dim: bool, opt: MotionConv3dEncoder2dOption
) -> MotionEncoder2d:
    return MotionConv3dEncoder2d(
        opt.in_channels,
        latent_dim,
        opt.conv_params,
        debug_show_dim=debug_show_dim,
    )


class MotionConv2dEncoder1d(MotionEncoder1d):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
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
        self.debug_show_dim = debug_show_dim

    def forward(
        self,
        x: Tensor,
        x_0: Tensor | None = None,
    ) -> Tensor:
        t = x.size(1)
        x = x.permute(0, 2, 1, 3)  # (b, c, t, h)
        x = self.enc(x)
        x = x.permute(0, 2, 1, 3)  # (b, t, c, h)
        assert x.size(1) == t
        if self.debug_show_dim:
            print(f"{self.__class__.__name__}", x.size())
        return x


class MotionConv3dEncoder2d(MotionEncoder2d):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()

        self.enc = ConvModule3d(
            in_channels,
            latent_dim,
            latent_dim,
            conv_params,
            transpose=False,
            debug_show_dim=debug_show_dim,
        )
        self.debug_show_dim = debug_show_dim

    def forward(
        self,
        x: Tensor,
        x_0: Tensor | None = None,
    ) -> Tensor:
        t = x.size(1)
        x = x.permute(0, 2, 1, 3, 4)  # (b, c, t, d, h)
        x = self.enc(x)
        x = x.permute(0, 2, 1, 3, 4)  # (b, t, c, d, h)
        assert x.size(1) == t
        if self.debug_show_dim:
            print(f"{self.__class__.__name__}", x.size())
        return x


@dataclass
class MotionGuidedEncoder1dOption(MotionNormalEncoder1dOption):
    pass


@dataclass
class MotionGuidedEncoder2dOption(MotionNormalEncoder2dOption):
    pass


def create_motion_guided_encoder1d(
    latent_dim: int, debug_show_dim: bool, opt: MotionGuidedEncoder1dOption
) -> MotionEncoder1d:
    return MotionGuidedEncoder1d(
        opt.in_channels,
        latent_dim,
        opt.conv_params,
        debug_show_dim=debug_show_dim,
    )


def create_motion_guided_encoder2d(
    latent_dim: int, debug_show_dim: bool, opt: MotionGuidedEncoder2dOption
) -> MotionEncoder2d:
    return MotionGuidedEncoder2d(
        opt.in_channels,
        latent_dim,
        opt.conv_params,
        debug_show_dim=debug_show_dim,
    )


class MotionGuidedEncoder1d(MotionNormalEncoder1d):
    def forward(
        self,
        x: Tensor,
        x_0: Tensor | None = None,
    ) -> Tensor:
        assert x_0 is not None
        x = super().forward(x, x_0)
        x_0 = self.enc(x_0).unsqueeze(1)
        x -= x_0
        if self.debug_show_dim:
            print(f"{self.__class__.__name__}", x.size())
        return x


class MotionGuidedEncoder2d(MotionNormalEncoder2d):
    def forward(
        self,
        x: Tensor,
        x_0: Tensor | None = None,
    ) -> Tensor:
        assert x_0 is not None
        x = super().forward(x, x_0)
        x_0 = self.enc(x_0).unsqueeze(1)
        x -= x_0
        if self.debug_show_dim:
            print(f"{self.__class__.__name__}", x.size())
        return x


@dataclass
class MotionTSNEncoder1dOption(MotionNormalEncoder1dOption):
    pass


@dataclass
class MotionTSNEncoder2dOption(MotionNormalEncoder2dOption):
    pass


def create_motion_tsn_encoder1d(
    latent_dim: int, debug_show_dim: bool, opt: MotionTSNEncoder1dOption
) -> MotionEncoder1d:
    return MotionTSNEncoder1d(
        opt.in_channels,
        latent_dim,
        opt.conv_params,
        debug_show_dim=debug_show_dim,
    )


def create_motion_tsn_encoder2d(
    latent_dim: int, debug_show_dim: bool, opt: MotionTSNEncoder2dOption
) -> MotionEncoder2d:
    return MotionTSNEncoder2d(
        opt.in_channels,
        latent_dim,
        opt.conv_params,
        debug_show_dim=debug_show_dim,
    )


class MotionTSNEncoder1d(MotionNormalEncoder1d):
    def forward(
        self,
        x: Tensor,
        x_0: Tensor | None = None,
    ) -> Tensor:
        # x: (b, t, c, h) - x_0: (b, s, h)
        x -= x_0.unsqueeze(1)
        x = super().forward(x, x_0)
        if self.debug_show_dim:
            print(f"{self.__class__.__name__}", x.size())
        return x


class MotionTSNEncoder2d(MotionNormalEncoder2d):
    def forward(
        self,
        x: Tensor,
        x_0: Tensor | None = None,
    ) -> Tensor:
        # x: (b, t, c, d, h) - x_0: (b, s, d, h)
        x -= x_0.unsqueeze(1)
        x = super().forward(x, x_0)
        if self.debug_show_dim:
            print(f"{self.__class__.__name__}", x.size())
        return x


if __name__ == "__main__":

    def test():
        from torch import randn

        from .rnn import ConvLSTM1d, ConvLSTM2d

        me_net = MotionRNNEncoder1d(
            1,
            16,
            [
                {"kernel_size": [3], "stride": [2], "padding": [1]},
                {"kernel_size": [3], "stride": [2], "padding": [1]},
                {"kernel_size": [3], "stride": [1], "padding": [1]},
            ],
            ConvLSTM1d(16, 2),
            debug_show_dim=True,
        )
        x = randn(8, 10, 1, 64)
        m = me_net(x)
        print("m input", x.size())
        print("m output", m.size())

        me_net = MotionRNNEncoder2d(
            1,
            16,
            [
                {"kernel_size": [3], "stride": [2], "padding": [1]},
                {"kernel_size": [3], "stride": [2], "padding": [1]},
                {"kernel_size": [3], "stride": [1], "padding": [1]},
            ],
            ConvLSTM2d(16, 2),
            debug_show_dim=True,
        )
        x = randn(8, 10, 1, 64, 64)
        m = me_net(x)
        print("m input", x.size())
        print("m output", m.size())

        me_net = MotionNormalEncoder1d(
            1,
            16,
            [
                {"kernel_size": [3], "stride": [2], "padding": [1]},
                {"kernel_size": [3], "stride": [2], "padding": [1]},
                {"kernel_size": [3], "stride": [1], "padding": [1]},
            ],
            debug_show_dim=True,
        )
        x = randn(8, 10, 1, 64)
        m = me_net(x)
        print("m input", x.size())
        print("m output", m.size())

        me_net = MotionNormalEncoder2d(
            1,
            16,
            [
                {"kernel_size": [3], "stride": [2], "padding": [1]},
                {"kernel_size": [3], "stride": [2], "padding": [1]},
                {"kernel_size": [3], "stride": [1], "padding": [1]},
            ],
            debug_show_dim=True,
        )
        x = randn(8, 10, 1, 64, 64)
        m = me_net(x)
        print("m input", x.size())
        print("m output", m.size())

        me_net = MotionConv2dEncoder1d(
            1,
            16,
            [
                {"kernel_size": [3], "stride": [1, 2], "padding": [1]},
                {"kernel_size": [3], "stride": [1, 2], "padding": [1]},
                {"kernel_size": [3], "stride": [1], "padding": [1]},
            ],
            debug_show_dim=True,
        )
        x = randn(8, 10, 1, 64)
        m = me_net(x)
        print("m input", x.size())
        print("m output", m.size())

        me_net = MotionConv3dEncoder2d(
            1,
            16,
            [
                {"kernel_size": [3], "stride": [1, 2, 2], "padding": [1]},
                {"kernel_size": [3], "stride": [1, 2, 2], "padding": [1]},
                {"kernel_size": [3], "stride": [1], "padding": [1]},
            ],
            debug_show_dim=True,
        )
        x = randn(8, 10, 1, 64, 64)
        m = me_net(x)
        print("m input", x.size())
        print("m output", m.size())

        me_net = MotionGuidedEncoder1d(
            1,
            16,
            [
                {"kernel_size": [3], "stride": [2], "padding": [1]},
                {"kernel_size": [3], "stride": [2], "padding": [1]},
                {"kernel_size": [3], "stride": [1], "padding": [1]},
            ],
            debug_show_dim=True,
        )
        x = randn(8, 10, 1, 64)
        m = me_net(x, randn(8, 1, 64))
        print("m input", x.size())
        print("m output", m.size())

        me_net = MotionGuidedEncoder2d(
            1,
            16,
            [
                {"kernel_size": [3], "stride": [2], "padding": [1]},
                {"kernel_size": [3], "stride": [2], "padding": [1]},
                {"kernel_size": [3], "stride": [1], "padding": [1]},
            ],
            debug_show_dim=True,
        )
        x = randn(8, 10, 1, 64, 64)
        m = me_net(x, randn(8, 1, 64, 64))
        print("m input", x.size())
        print("m output", m.size())

        me_net = MotionTSNEncoder1d(
            1,
            16,
            [
                {"kernel_size": [3], "stride": [2], "padding": [1]},
                {"kernel_size": [3], "stride": [2], "padding": [1]},
                {"kernel_size": [3], "stride": [1], "padding": [1]},
            ],
            debug_show_dim=True,
        )
        x = randn(8, 10, 1, 64)
        m = me_net(x, randn(8, 1, 64))
        print("m input", x.size())
        print("m output", m.size())

        me_net = MotionTSNEncoder2d(
            1,
            16,
            [
                {"kernel_size": [3], "stride": [2], "padding": [1]},
                {"kernel_size": [3], "stride": [2], "padding": [1]},
                {"kernel_size": [3], "stride": [1], "padding": [1]},
            ],
            debug_show_dim=True,
        )
        x = randn(8, 10, 1, 64, 64)
        m = me_net(x, randn(8, 1, 64, 64))
        print("m input", x.size())
        print("m output", m.size())

    test()
