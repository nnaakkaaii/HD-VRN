from torch import nn, Tensor


class ResNetBranch(nn.Module):
    def __init__(self, *args: nn.Module) -> None:
        super().__init__()
        self.net = nn.Sequential(*args)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x) + x
