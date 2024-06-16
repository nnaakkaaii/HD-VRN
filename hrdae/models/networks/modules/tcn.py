from pytorch_tcn import TCN
from torch import Tensor, nn


class TCN1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int | list[int],
        kernel_size: int,
        image_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if isinstance(hidden_dim, int):
            self.c = hidden_dim
            hidden_dim *= image_size
        elif isinstance(hidden_dim, list):
            self.c = hidden_dim[-1]
            hidden_dim = [h * image_size for h in hidden_dim]
        else:
            raise ValueError("hidden_dim must be int or list[int]")

        self.tcn = TCN(
            num_inputs=in_channels * image_size,
            num_channels=hidden_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            causal=True,
            input_shape="NLC",
        )

    def forward(self, x: Tensor) -> Tensor:
        b, t, c, h = x.size()
        x = x.reshape(b, t, c * h)
        x = self.tcn(x)
        x = x.reshape(b, t, self.c, h)
        return x


class TCN2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int | list[int],
        kernel_size: int,
        image_size: tuple[int, int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        s = image_size[0] * image_size[1]
        if isinstance(hidden_dim, int):
            self.c = hidden_dim
            hidden_dim *= s
        elif isinstance(hidden_dim, list):
            self.c = hidden_dim[-1]
            hidden_dim = [h * s for h in hidden_dim]
        else:
            raise ValueError("hidden_dim must be int or list[int]")

        self.tcn = TCN(
            num_inputs=in_channels * s,
            num_channels=hidden_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            causal=True,
            input_shape="NLC",
        )

    def forward(self, x: Tensor) -> Tensor:
        b, t, c, h, w = x.size()
        print(b, t, c, h, w)
        x = x.reshape(b, t, c * h * w)
        x = self.tcn(x)
        x = x.reshape(b, t, self.c, h, w)
        return x
