from torch import nn

from .autoencoder import AutoEncoder2dNetworkOption, create_autoencoder2d
from .discriminator import Discriminator3dOption, create_discriminator3d
from .fb_autoencoder import (FiveBranchAutoencoder3dOption,
                             create_fb_autoencoder3d)
from .hrd3d_vrn import HRD3DVRNOption, create_hrd3dvrn
from .option import NetworkOption
from .rd3d_vrn import RD3DVRNOption, create_rd3dvrn
from .straight_vrn import SVRNOption, create_svrn


def create_network(opt: NetworkOption) -> nn.Module:
    if isinstance(opt, AutoEncoder2dNetworkOption) and type(opt) is AutoEncoder2dNetworkOption:
        return create_autoencoder2d(opt)
    if isinstance(opt, Discriminator3dOption) and type(opt) is Discriminator3dOption:
        return create_discriminator3d(opt)
    if isinstance(opt, FiveBranchAutoencoder3dOption) and type(opt) is FiveBranchAutoencoder3dOption:
        return create_fb_autoencoder3d(opt)
    if isinstance(opt, HRD3DVRNOption) and type(opt) is HRD3DVRNOption:
        return create_hrd3dvrn(opt)
    if isinstance(opt, RD3DVRNOption) and type(opt) is RD3DVRNOption:
        return create_rd3dvrn(opt)
    if isinstance(opt, SVRNOption) and type(opt) is SVRNOption:
        return create_svrn(opt)
    raise NotImplementedError(f"network {opt.__class__.__name__} not implemented")
