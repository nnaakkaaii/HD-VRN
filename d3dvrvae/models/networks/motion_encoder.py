from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field

from omegaconf import MISSING
from torch import Tensor, nn

from .modules import ConvModule2d
from .rnn import RNN2d, RNN2dOption, create_rnn2d


@dataclass
class MotionEncoder2dOption:
    pass


def create_motion_encoder2d(
    latent_dim: int, debug_show_dim: bool, opt: MotionEncoder2dOption
) -> "MotionEncoder2d":
    if isinstance(opt, MotionRNNEncoder2dOption):
        return create_motion_rnn_encoder2d(latent_dim, debug_show_dim, opt)
    raise NotImplementedError(f"{opt.__class__.__name__} not implemented")


class MotionEncoder2d(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(
        self,
        x: Tensor,
        x_0: Tensor | None = None,
    ) -> Tensor:
        pass


@dataclass
class MotionRNNEncoder2dOption(MotionEncoder2dOption):
    in_channels: int
    conv_params: list[dict[str, int]] = field(
        default_factory=lambda: [{"kernel_size": 3, "stride": 2, "padding": 1}] * 3,
    )
    rnn2d: RNN2dOption = MISSING


def create_motion_rnn_encoder2d(
    latent_dim: int, debug_show_dim: bool, opt: MotionRNNEncoder2dOption
) -> MotionEncoder2d:
    return MotionRNNEncoder2d(
        opt.in_channels,
        latent_dim,
        opt.conv_params,
        create_rnn2d(latent_dim, opt.rnn2d),
        debug_show_dim=debug_show_dim,
    )


class MotionRNNEncoder2d(MotionEncoder2d):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        conv_params: list[dict[str, int]],
        rnn2d: RNN2d,
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

    def forward(
        self,
        x: Tensor,
        x_0: Tensor | None = None,
    ) -> Tensor:
        b, t, c, h, w = x.size()
        x = x.view(b * t, c, h, w)
        x = self.enc(x)
        _, c, h, w = x.size()
        x = x.view(b, t, c, h, w)
        x, _ = self.rnn(x)
        if self.debug_show_dim:
            print(f"{self.__class__.__name__}", x.size())
        return x


if __name__ == "__main__":

    def test():
        from torch import randn

        from .rnn import ConvLSTM2d

        me_net = MotionRNNEncoder2d(
            1,
            16,
            [
                {"kernel_size": 3, "stride": 2, "padding": 1},
                {"kernel_size": 3, "stride": 2, "padding": 1},
                {"kernel_size": 3, "stride": 1, "padding": 1},
            ],
            ConvLSTM2d(16, 2),
            debug_show_dim=True,
        )
        x = randn(8, 10, 1, 64, 64)
        m = me_net(x)
        print("m input", x.size())
        print("m output", m.size())

    test()
