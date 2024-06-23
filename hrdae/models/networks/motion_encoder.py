from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field

from omegaconf import MISSING
from torch import Tensor, nn

from .modules import (
    ConvModule1d,
    ConvModule2d,
    ConvModule3d,
    PixelWiseConv1d,
    PixelWiseConv2d,
    PixelWiseConv3d,
)
from .rnn import RNN1d, RNN1dOption, RNN2d, RNN2dOption, create_rnn1d, create_rnn2d


@dataclass
class MotionEncoder1dOption:
    # num slices * (1 (agg=diff|none) | 2 (phase=0, t) | 3 (phase=all))
    in_channels: int
    hidden_channels: int
    conv_params: list[dict[str, list[int]]] = field(
        default_factory=lambda: [{"kernel_size": [3], "stride": [2], "padding": [1]}]
        * 3,
    )


@dataclass
class MotionEncoder2dOption:
    in_channels: int
    hidden_channels: int
    conv_params: list[dict[str, list[int]]] = field(
        default_factory=lambda: [{"kernel_size": [3], "stride": [2], "padding": [1]}]
        * 3,
    )


def create_motion_encoder1d(
    latent_dim: int, debug_show_dim: bool, opt: MotionEncoder1dOption
) -> "MotionEncoder1d":
    if (
        isinstance(opt, MotionRNNEncoder1dOption)
        and type(opt) is MotionRNNEncoder1dOption
    ):
        return MotionRNNEncoder1d(
            opt.in_channels,
            opt.hidden_channels,
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
            opt.in_channels,
            opt.hidden_channels,
            latent_dim,
            opt.conv_params,
            debug_show_dim,
        )
    if (
        isinstance(opt, MotionConv2dEncoder1dOption)
        and type(opt) is MotionConv2dEncoder1dOption
    ):
        return MotionConv2dEncoder1d(
            opt.in_channels,
            opt.hidden_channels,
            latent_dim,
            opt.conv_params,
            debug_show_dim,
        )
    if (
        isinstance(opt, MotionGuidedEncoder1dOption)
        and type(opt) is MotionGuidedEncoder1dOption
    ):
        return MotionGuidedEncoder1d(
            opt.in_channels,
            opt.hidden_channels,
            latent_dim,
            opt.conv_params,
            debug_show_dim,
        )
    raise NotImplementedError(f"{opt.__class__.__name__} not implemented")


def create_motion_encoder2d(
    latent_dim: int, debug_show_dim: bool, opt: MotionEncoder2dOption
) -> "MotionEncoder2d":
    if (
        isinstance(opt, MotionRNNEncoder2dOption)
        and type(opt) is MotionRNNEncoder2dOption
    ):
        return MotionRNNEncoder2d(
            opt.in_channels,
            opt.hidden_channels,
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
            opt.in_channels,
            opt.hidden_channels,
            latent_dim,
            opt.conv_params,
            debug_show_dim,
        )
    if (
        isinstance(opt, MotionConv3dEncoder2dOption)
        and type(opt) is MotionConv3dEncoder2dOption
    ):
        return MotionConv3dEncoder2d(
            opt.in_channels,
            opt.hidden_channels,
            latent_dim,
            opt.conv_params,
            debug_show_dim,
        )
    if (
        isinstance(opt, MotionGuidedEncoder2dOption)
        and type(opt) is MotionGuidedEncoder2dOption
    ):
        return MotionGuidedEncoder2d(
            opt.in_channels,
            opt.hidden_channels,
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
        hidden_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()

        self.cnn = ConvModule1d(
            in_channels,
            hidden_channels,
            hidden_channels,
            conv_params,
            transpose=False,
            act_norm=True,
            debug_show_dim=debug_show_dim,
        )
        self.bottleneck = PixelWiseConv1d(
            hidden_channels,
            latent_dim,
            act_norm=False,
        )
        self.debug_show_dim = debug_show_dim

    def forward(
        self,
        x: Tensor,
        x_0: Tensor | None = None,
    ) -> Tensor:
        b, t, c, h = x.size()
        x = x.reshape(b * t, c, h)
        y = self.cnn(x)
        z = self.bottleneck(y)
        _, c, h = z.size()
        z = z.reshape(b, t, c, h)
        if self.debug_show_dim:
            print(f"{self.__class__.__name__}", z.size())
        return z


class MotionNormalEncoder2d(MotionEncoder2d):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()

        self.cnn = ConvModule2d(
            in_channels,
            hidden_channels,
            hidden_channels,
            conv_params,
            transpose=False,
            act_norm=True,
            debug_show_dim=debug_show_dim,
        )
        self.bottleneck = PixelWiseConv2d(
            hidden_channels,
            latent_dim,
            act_norm=False,
        )
        self.debug_show_dim = debug_show_dim

    def forward(
        self,
        x: Tensor,
        x_0: Tensor | None = None,
    ) -> Tensor:
        b, t, c, d, h = x.size()
        x = x.reshape(b * t, c, d, h)
        y = self.cnn(x)
        z = self.bottleneck(y)
        _, c, d, h = z.size()
        z = z.reshape(b, t, c, d, h)
        if self.debug_show_dim:
            print(f"{self.__class__.__name__}", z.size())
        return z


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
        hidden_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        rnn: RNN1d,
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            hidden_channels,
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
        z = super().forward(x, x_0)
        z, _ = self.rnn(z)
        if self.debug_show_dim:
            print(f"{self.__class__.__name__}", z.size())
        return z


class MotionRNNEncoder2d(MotionNormalEncoder2d):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        rnn: RNN2d,
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            hidden_channels,
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
        z = super().forward(x, x_0)
        z, _ = self.rnn(z)
        if self.debug_show_dim:
            print(f"{self.__class__.__name__}", z.size())
        return z


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
        hidden_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()

        self.cnn = ConvModule2d(
            in_channels,
            hidden_channels,
            hidden_channels,
            conv_params,
            transpose=False,
            act_norm=True,
            debug_show_dim=debug_show_dim,
        )
        self.bottleneck = PixelWiseConv2d(
            hidden_channels,
            latent_dim,
            act_norm=False,
        )
        self.debug_show_dim = debug_show_dim

    def forward(
        self,
        x: Tensor,
        x_0: Tensor | None = None,
    ) -> Tensor:
        t = x.size(1)
        x = x.permute(0, 2, 1, 3)  # (b, c, t, h)
        y = self.cnn(x)
        z = self.bottleneck(y)
        z = z.permute(0, 2, 1, 3)  # (b, t, c, h)
        assert z.size(1) == t
        if self.debug_show_dim:
            print(f"{self.__class__.__name__}", z.size())
        return z


class MotionConv3dEncoder2d(MotionEncoder2d):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, list[int]]],
        debug_show_dim: bool = False,
    ) -> None:
        super().__init__()

        self.cnn = ConvModule3d(
            in_channels,
            hidden_channels,
            hidden_channels,
            conv_params,
            transpose=False,
            act_norm=True,
            debug_show_dim=debug_show_dim,
        )
        self.bottleneck = PixelWiseConv3d(
            hidden_channels,
            latent_dim,
            act_norm=False,
        )
        self.debug_show_dim = debug_show_dim

    def forward(
        self,
        x: Tensor,
        x_0: Tensor | None = None,
    ) -> Tensor:
        t = x.size(1)
        x = x.permute(0, 2, 1, 3, 4)  # (b, c, t, d, h)
        y = self.cnn(x)
        z = self.bottleneck(y)
        z = z.permute(0, 2, 1, 3, 4)  # (b, t, c, d, h)
        assert z.size(1) == t
        if self.debug_show_dim:
            print(f"{self.__class__.__name__}", z.size())
        return z


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
        z = super().forward(x, x_0)
        z_0 = self.bottleneck(self.cnn(x_0)).unsqueeze(1)
        z -= z_0
        if self.debug_show_dim:
            print(f"{self.__class__.__name__}", z.size())
        return z


class MotionGuidedEncoder2d(MotionNormalEncoder2d):
    def forward(
        self,
        x: Tensor,
        x_0: Tensor | None = None,
    ) -> Tensor:
        assert x_0 is not None
        z = super().forward(x, x_0)
        z_0 = self.bottleneck(self.cnn(x_0)).unsqueeze(1)
        z -= z_0
        if self.debug_show_dim:
            print(f"{self.__class__.__name__}", z.size())
        return z
