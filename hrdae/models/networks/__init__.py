from torch import nn

from .autoencoder import (
    AEEncoder2dNetworkOption,
    AEEncoder3dNetworkOption,
    AutoEncoder2dNetworkOption,
    AutoEncoder3dNetworkOption,
    create_ae_encoder2d,
    create_ae_encoder3d,
    create_autoencoder2d,
    create_autoencoder3d,
)
from .discriminator import (
    Discriminator2dOption,
    Discriminator3dOption,
    create_discriminator2d,
    create_discriminator3d,
)
from .hr_dae import HRDAE2dOption, HRDAE3dOption, create_hrdae2d, create_hrdae3d
from .option import NetworkOption
from .r_ae import RAE2dOption, RAE3dOption, create_rae2d, create_rae3d
from .r_dae import RDAE2dOption, RDAE3dOption, create_rdae2d, create_rdae3d


def create_network(out_channels: int, opt: NetworkOption) -> nn.Module:
    if isinstance(opt, Discriminator2dOption) and type(opt) is Discriminator2dOption:
        return create_discriminator2d(opt)
    if isinstance(opt, Discriminator3dOption) and type(opt) is Discriminator3dOption:
        return create_discriminator3d(opt)
    if (
        isinstance(opt, AEEncoder2dNetworkOption)
        and type(opt) is AEEncoder2dNetworkOption
    ):
        return create_ae_encoder2d(opt)
    if (
        isinstance(opt, AEEncoder3dNetworkOption)
        and type(opt) is AEEncoder3dNetworkOption
    ):
        return create_ae_encoder3d(opt)
    if (
        isinstance(opt, AutoEncoder2dNetworkOption)
        and type(opt) is AutoEncoder2dNetworkOption
    ):
        return create_autoencoder2d(out_channels, opt)
    if (
        isinstance(opt, AutoEncoder3dNetworkOption)
        and type(opt) is AutoEncoder3dNetworkOption
    ):
        return create_autoencoder3d(out_channels, opt)
    if isinstance(opt, HRDAE2dOption) and type(opt) is HRDAE2dOption:
        return create_hrdae2d(out_channels, opt)
    if isinstance(opt, HRDAE3dOption) and type(opt) is HRDAE3dOption:
        return create_hrdae3d(out_channels, opt)
    if isinstance(opt, RAE2dOption) and type(opt) is RAE2dOption:
        return create_rae2d(out_channels, opt)
    if isinstance(opt, RAE3dOption) and type(opt) is RAE3dOption:
        return create_rae3d(out_channels, opt)
    if isinstance(opt, RDAE2dOption) and type(opt) is RDAE2dOption:
        return create_rdae2d(out_channels, opt)
    if isinstance(opt, RDAE3dOption) and type(opt) is RDAE3dOption:
        return create_rdae3d(out_channels, opt)
    raise NotImplementedError(f"network {opt.__class__.__name__} not implemented")


__all__ = [
    "Discriminator2dOption",
    "Discriminator3dOption",
    "AEEncoder2dNetworkOption",
    "AEEncoder3dNetworkOption",
    "AutoEncoder2dNetworkOption",
    "AutoEncoder3dNetworkOption",
    "HRDAE2dOption",
    "HRDAE3dOption",
    "RAE2dOption",
    "RAE3dOption",
    "RDAE2dOption",
    "RDAE3dOption",
    "NetworkOption",
    "create_network",
]
