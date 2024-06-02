from torch import nn

from .autoencoder import create_autoencoder2d, AutoEncoder2dNetworkOption
from .option import NetworkOption


def create_network(opt: NetworkOption) -> nn.Module:
    if isinstance(opt, AutoEncoder2dNetworkOption):
        return create_autoencoder2d(opt)
    raise NotImplementedError(f"network {opt.__class__.__name__} not implemented")
