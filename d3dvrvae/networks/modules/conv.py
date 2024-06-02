from torch import nn, Tensor


class BasicConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 transpose: bool = False,
                 act_norm: bool = False) -> None:
        super().__init__()
        self.act_norm = act_norm

        if not transpose:
            self.conv = nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=padding,
                                  )
        else:
            self.conv = nn.ConvTranspose2d(in_channels,
                                           out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=padding,
                                           output_padding=stride // 2,
                                           )

        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 transpose: bool = False,
                 act_norm: bool = True,
                 ) -> None:
        super().__init__()
        if stride == 1:
            transpose = False

        self.conv = BasicConv2d(in_channels,
                                out_channels,
                                kernel_size=3,
                                stride=stride,
                                padding=1,
                                transpose=transpose,
                                act_norm=act_norm,
                                )

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv(x)
        return y
