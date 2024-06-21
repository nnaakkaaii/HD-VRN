from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field

from omegaconf import MISSING
from torch import Tensor, nn

from .modules import ConvModule1d, ConvModule2d, ConvModule3d
from .rnn import RNN1d, RNN1dOption, RNN2d, RNN2dOption, create_rnn1d, create_rnn2d


@dataclass
class MotionEncoder1dOption:
    in_channels: int
    conv_params: list[dict[str, list[int]]] = field(
        default_factory=lambda: [{"kernel_size": [3], "stride": [2], "padding": [1]}]
        * 3,
    )
    aggregation: str = "concat"  # diff or concat
    phase: str = "none"  # all, 0, t, none


@dataclass
class MotionEncoder2dOption:
    in_channels: int
    conv_params: list[dict[str, list[int]]] = field(
        default_factory=lambda: [{"kernel_size": [3], "stride": [2], "padding": [1]}]
        * 3,
    )
    aggregation: str = "concat"  # diff or concat
    phase: str = "none"  # all, 0, t, none


def check_in_channels(
    in_channels: int, opt: MotionEncoder1dOption | MotionEncoder2dOption
) -> None:
    if in_channels == 1:
        return
    assert type(opt) is not MotionGuidedEncoder1dOption
    assert type(opt) is not MotionGuidedEncoder2dOption


def create_motion_encoder1d(
    latent_dim: int, debug_show_dim: bool, opt: MotionEncoder1dOption
) -> "MotionEncoder1d":
    assert opt.aggregation in ["diff", "concat"]
    assert opt.phase in ["all", "0", "t", "none"]
    if opt.aggregation == "diff":
        assert opt.phase in ["0", "t"]

    in_channels = opt.in_channels
    if opt.aggregation == "concat":
        if opt.phase in ["all"]:
            in_channels *= 3
        if opt.phase in ["0", "t"]:
            in_channels *= 2

    if (
        isinstance(opt, MotionRNNEncoder1dOption)
        and type(opt) is MotionRNNEncoder1dOption
    ):
        return MotionRNNEncoder1d(
            in_channels,
            latent_dim,
            opt.conv_params,
            create_rnn1d(latent_dim, opt.rnn),
            debug_show_dim,
        )
    if (
        isinstance(opt, MotionNormalEncoder1dOption)
        and type(opt) is MotionNormalEncoder1dOption
    ):
        return MotionNormalEncoder1d(
            in_channels,
            latent_dim,
            opt.conv_params,
            debug_show_dim,
        )
    if (
        isinstance(opt, MotionConv2dEncoder1dOption)
        and type(opt) is MotionConv2dEncoder1dOption
    ):
        return MotionConv2dEncoder1d(
            in_channels,
            latent_dim,
            opt.conv_params,
            debug_show_dim,
        )
    if (
        isinstance(opt, MotionGuidedEncoder1dOption)
        and type(opt) is MotionGuidedEncoder1dOption
    ):
        return MotionGuidedEncoder1d(
            in_channels,
            latent_dim,
            opt.conv_params,
            debug_show_dim,
        )
    raise NotImplementedError(f"{opt.__class__.__name__} not implemented")


def create_motion_encoder2d(
    latent_dim: int, debug_show_dim: bool, opt: MotionEncoder2dOption
) -> "MotionEncoder2d":
    assert opt.aggregation in ["diff", "concat"]
    assert opt.phase in ["all", "0", "t", "none"]
    if opt.aggregation == "diff":
        assert opt.phase in ["0", "t"]

    in_channels = opt.in_channels
    if opt.aggregation == "concat":
        if opt.phase in ["all"]:
            in_channels *= 3
        if opt.phase in ["0", "t"]:
            in_channels *= 2

    if (
        isinstance(opt, MotionRNNEncoder2dOption)
        and type(opt) is MotionRNNEncoder2dOption
    ):
        return MotionRNNEncoder2d(
            in_channels,
            latent_dim,
            opt.conv_params,
            create_rnn2d(latent_dim, opt.rnn),
            debug_show_dim,
        )
    if (
        isinstance(opt, MotionNormalEncoder2dOption)
        and type(opt) is MotionNormalEncoder2dOption
    ):
        return MotionNormalEncoder2d(
            in_channels,
            latent_dim,
            opt.conv_params,
            debug_show_dim,
        )
    if (
        isinstance(opt, MotionConv3dEncoder2dOption)
        and type(opt) is MotionConv3dEncoder2dOption
    ):
        return MotionConv3dEncoder2d(
            in_channels,
            latent_dim,
            opt.conv_params,
            debug_show_dim,
        )
    if (
        isinstance(opt, MotionGuidedEncoder2dOption)
        and type(opt) is MotionGuidedEncoder2dOption
    ):
        return MotionGuidedEncoder2d(
            in_channels,
            latent_dim,
            opt.conv_params,
            debug_show_dim,
        )
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
    pass


@dataclass
class MotionNormalEncoder2dOption(MotionEncoder2dOption):
    pass


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
        x = x.reshape(b * t, c, h)
        x = self.enc(x)
        _, c, h = x.size()
        x = x.reshape(b, t, c, h)
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
        x = x.reshape(b * t, c, d, h)
        x = self.enc(x)
        _, c, d, h = x.size()
        x = x.reshape(b, t, c, d, h)
        if self.debug_show_dim:
            print(f"{self.__class__.__name__}", x.size())
        return x


@dataclass
class MotionRNNEncoder1dOption(MotionEncoder1dOption):
    rnn: RNN1dOption = MISSING


@dataclass
class MotionRNNEncoder2dOption(MotionEncoder2dOption):
    rnn: RNN2dOption = MISSING


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
class MotionConv2dEncoder1dOption(MotionEncoder1dOption):
    pass


@dataclass
class MotionConv3dEncoder2dOption(MotionEncoder2dOption):
    pass


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
class MotionGuidedEncoder1dOption(MotionEncoder1dOption):
    pass


@dataclass
class MotionGuidedEncoder2dOption(MotionEncoder2dOption):
    pass


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
