from abc import ABCMeta, abstractmethod
from dataclasses import dataclass

from torch import nn, Tensor

from .option import NetworkOption
from . import modules as mdl


class RNN2d(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(
            self,
            x: Tensor,
            last_states: list[tuple[Tensor, Tensor]] | Tensor | None = None,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]] | Tensor | None]:
        pass


@dataclass
class RNN2dOption(NetworkOption):
    pass


def create_rnn2d(latent_dim: int, opt: RNN2dOption) -> RNN2d:
    if isinstance(opt, NoRNN2dOption):
        return create_no_rnn2d(latent_dim, opt)
    if isinstance(opt, ConvLSTM2dOption):
        return create_conv_lstm2d(latent_dim, opt)
    if isinstance(opt, GRU2dOption):
        return create_gru2d(latent_dim, opt)
    if isinstance(opt, TCN2dOption):
        return create_tcn2d(latent_dim, opt)
    raise NotImplementedError(f"{opt.__class__.__name__} not implemented")


@dataclass
class NoRNN2dOption(RNN2dOption):
    pass


def create_no_rnn2d(latent_dim: int, opt: NoRNN2dOption) -> RNN2d:
    return NoRNN2d()


class NoRNN2d(RNN2d):
    def forward(
            self,
            x: Tensor,
            last_states: list[tuple[Tensor, Tensor]] | Tensor | None = None,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]] | Tensor | None]:
        return x, last_states


@dataclass
class ConvLSTM2dOption(RNN2dOption):
    num_layers: int = 3


def create_conv_lstm2d(
        latent_dim: int, opt: ConvLSTM2dOption
) -> RNN2d:
    return ConvLSTM2d(latent_dim, opt.num_layers)


class ConvLSTM2d(RNN2d):
    def __init__(
            self,
            latent_dim: int,
            num_layers: int,
    ) -> None:
        super().__init__()
        self.rnn = mdl.ConvLSTM2d(
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
class GRU2dOption(RNN2dOption):
    num_layers: int = 3
    image_size: tuple[int, int] = (64, 64)


def create_gru2d(latent_dim: int, opt: GRU2dOption) -> RNN2d:
    return GRU2d(latent_dim, opt.num_layers, opt.image_size)


class GRU2d(RNN2d):
    def __init__(
            self,
            latent_dim: int,
            num_layers: int,
            image_size: tuple[int, int],
    ) -> None:
        super().__init__()
        self.rnn = mdl.GRU2d(
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
class TCN2dOption(RNN2dOption):
    num_layers: int = 3
    image_size: tuple[int, int] = (64, 64)
    kernel_size: int = 3
    dropout: float = 0.0


def create_tcn2d(latent_dim: int, opt: TCN2dOption) -> RNN2d:
    return TCN2d(
        latent_dim, opt.num_layers, opt.image_size, opt.kernel_size, opt.dropout
    )


class TCN2d(RNN2d):
    def __init__(
            self,
            latent_dim: int,
            num_layers: int,
            image_size: tuple[int, int],
            kernel_size: int,
            dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.rnn = mdl.TCN2d(
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
