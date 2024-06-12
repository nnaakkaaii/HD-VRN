# Licensed under the Apache License, Version 2.0 (the "License");
# Hierarchical Recurrent Disentangled 3D Video Reconstruction Network (HRD3D-VRN)

from dataclasses import dataclass, field

from torch import nn, Tensor

from .option import NetworkOption
from .modules import HierarchicalConvEncoder3d, HierarchicalConvDecoder3d, IdenticalConvBlock3d, ResNetBranch


@dataclass
class HRD3DVRNOption(NetworkOption):
    pass


def create_hrd3dvrn(opt: HRD3DVRNOption) -> nn.Module:
    pass


class ContentEncoder3d(HierarchicalConvEncoder3d):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 conv_params: list[dict[str, int]],
                 debug_show_dim: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            latent_dim,
            latent_dim,
            conv_params,
            debug_show_dim,
        )


class Decoder3d(nn.Module):
    def __init__(
            self,
            out_channels: int,
            latent_dim: int,
            conv_params: list[dict[str, int]],
            debug_show_dim: bool = False,
    ) -> None:
        super().__init__()

        self.dec = HierarchicalConvDecoder3d(
            latent_dim,
            out_channels,
            latent_dim,
            conv_params,
            debug_show_dim,
        )
        # motion guided connection
        # (Mutual Suppression Network for Video Prediction using Disentangled Features)
        self.mgc = nn.ModuleList()
        for _ in conv_params:
            self.mgc.append(nn.Sequential(
                ResNetBranch(
                    IdenticalConvBlock3d(latent_dim, latent_dim),
                    IdenticalConvBlock3d(latent_dim, latent_dim, act_norm=False),
                ),
                nn.GroupNorm(2, latent_dim),
                nn.LeakyReLU(0.2, inplace=True),
            ))

    def forward(self, x: Tensor, hs: list[Tensor]) -> Tensor:
        assert len(self.mgc) == len(hs)
        for i, (mgc, h) in enumerate(zip(self.mgc, hs)):
            hs[i] = mgc(h)

        return self.dec(x, hs)


if __name__ == "__main__":
    def test():
        from torch import randn

        ce_net = ContentEncoder3d(
            1, 16,
            [
                {"kernel_size": 3, "stride": 2, "padding": 1},
                {"kernel_size": 3, "stride": 2, "padding": 1},
                {"kernel_size": 3, "stride": 1, "padding": 1},
            ],
            debug_show_dim=True,
        )
        c, cs = ce_net(randn(8, 1, 64, 64, 64))
        for i, c_ in enumerate(cs):
            print(f"c{i}", c_.shape)
        print("c", c.shape)

        d_net = Decoder3d(
            1, 2*16,
            [
                {"kernel_size": 3, "stride": 2, "padding": 1},
                {"kernel_size": 3, "stride": 2, "padding": 1},
            ],
            debug_show_dim=True,
        )
        d = d_net(randn(8, 32, 16, 16, 16), [c.repeat(1, 2, 1, 1, 1) for c in cs[::-1]])
        print(d.shape)

    test()
