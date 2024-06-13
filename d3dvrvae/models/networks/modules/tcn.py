from pytorch_tcn import TCN
from torch import Tensor, nn


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
        x = x.view(b, t, c * h * w)
        x = self.tcn(x)
        x = x.view(b, t, self.c, h, w)
        return x


if __name__ == "__main__":

    def test():
        from torch import randn

        x = randn((32, 10, 64, 4, 4))
        tcn2d = TCN2d(64, [256, 256, 256], 3, (4, 4))
        x = tcn2d(x)
        print(x.size())

    test()
