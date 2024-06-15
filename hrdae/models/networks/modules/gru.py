from torch import Tensor, nn


class GRU1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        image_size: int,
    ) -> None:
        super().__init__()

        self.c = hidden_dim
        hidden_dim *= image_size

        self.rnn = nn.GRU(
            input_size=in_channels * image_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(
        self, x: Tensor, last_states: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        b, t, c, h = x.size()
        x = x.reshape(b, t, c * h)
        y, _last_states = self.rnn(x, last_states)
        y = y.reshape(b, t, self.c, h)
        return y, _last_states


class GRU2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        num_layers: int,
        image_size: list[int],
    ) -> None:
        super().__init__()

        assert len(image_size) == 2
        s = image_size[0] * image_size[1]
        self.c = hidden_dim
        hidden_dim *= s

        self.rnn = nn.GRU(
            input_size=in_channels * s,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(
        self, x: Tensor, last_states: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        b, t, c, h, w = x.size()
        x = x.reshape(b, t, c * h * w)
        y, _last_states = self.rnn(x, last_states)
        y = y.reshape(b, t, self.c, h, w)
        return y, _last_states
