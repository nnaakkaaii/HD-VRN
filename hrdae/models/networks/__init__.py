from torch import nn

from .autoencoder import AutoEncoder2dNetworkOption, create_autoencoder2d
from .discriminator import (
    Discriminator2dOption,
    Discriminator3dOption,
    create_discriminator2d,
    create_discriminator3d,
)
from .fb_autoencoder import (
    FiveBranchAutoencoder2d,
    FiveBranchAutoencoder2dOption,
    FiveBranchAutoencoder3dOption,
    create_fb_autoencoder3d,
)
from .hr_dae import HRDAE2dOption, HRDAE3dOption, create_hrdae2d, create_hrdae3d
from .option import NetworkOption
from .r_ae import RAE2dOption, RAE3dOption, create_rae2d, create_rae3d
from .r_dae import RDAE2dOption, RDAE3dOption, create_rdae2d, create_rdae3d


def create_network(opt: NetworkOption) -> nn.Module:
    if (
        isinstance(opt, AutoEncoder2dNetworkOption)
        and type(opt) is AutoEncoder2dNetworkOption
    ):
        return create_autoencoder2d(opt)
    if isinstance(opt, Discriminator3dOption) and type(opt) is Discriminator3dOption:
        return create_discriminator3d(opt)
    if (
        isinstance(opt, FiveBranchAutoencoder3dOption)
        and type(opt) is FiveBranchAutoencoder3dOption
    ):
        return create_fb_autoencoder3d(opt)
    if isinstance(opt, HRDAE2dOption) and type(opt) is HRDAE2dOption:
        return create_hrdae2d(opt)
    if isinstance(opt, HRDAE3dOption) and type(opt) is HRDAE3dOption:
        return create_hrdae3d(opt)
    if isinstance(opt, RAE2dOption) and type(opt) is RAE2dOption:
        return create_rae2d(opt)
    if isinstance(opt, RAE3dOption) and type(opt) is RAE3dOption:
        return create_rae3d(opt)
    if isinstance(opt, RDAE2dOption) and type(opt) is RDAE2dOption:
        return create_rdae2d(opt)
    if isinstance(opt, RDAE3dOption) and type(opt) is RDAE3dOption:
        return create_rdae3d(opt)
    raise NotImplementedError(f"network {opt.__class__.__name__} not implemented")
