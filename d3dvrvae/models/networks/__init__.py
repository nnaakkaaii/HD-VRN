from torch import nn

from .autoencoder import AutoEncoder2dNetworkOption, create_autoencoder2d
from .discriminator import Discriminator3dOption, create_discriminator3d
from .fb_autoencoder import (FiveBranchAutoencoder3dOption,
                             create_fb_autoencoder3d)
from .hrd3d_vrn import HRD3DVRNOption, create_hrd3dvrn
from .option import NetworkOption


def create_network(opt: NetworkOption) -> nn.Module:
    if isinstance(opt, AutoEncoder2dNetworkOption):
        return create_autoencoder2d(opt)
    if isinstance(opt, HRD3DVRNOption):
        return create_hrd3dvrn(opt)
    if isinstance(opt, Discriminator3dOption):
        return create_discriminator3d(opt)
    if isinstance(opt, FiveBranchAutoencoder3dOption):
        return create_fb_autoencoder3d(opt)
    raise NotImplementedError(f"network {opt.__class__.__name__} not implemented")
