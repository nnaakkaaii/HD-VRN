from torch import nn

from .aggregator import create_aggregator2d, create_aggregator3d
from .conv_block import (
    ConvBlock1d,
    ConvBlock2d,
    ConvBlock3d,
    ConvModule1d,
    ConvModule2d,
    ConvModule3d,
    IdenticalConvBlock1d,
    IdenticalConvBlock2d,
    IdenticalConvBlock3d,
    IdenticalConvBlockConvParams,
    PixelWiseConv1d,
    PixelWiseConv2d,
    PixelWiseConv3d,
)
from .conv_decoder import (
    HierarchicalConvDecoder1d,
    HierarchicalConvDecoder2d,
    HierarchicalConvDecoder3d,
)
from .conv_encoder import (
    HierarchicalConvEncoder1d,
    HierarchicalConvEncoder2d,
    HierarchicalConvEncoder3d,
)
from .conv_lstm import ConvLSTM1d, ConvLSTM2d
from .gru import GRU1d, GRU2d
from .resnet_block import ResNetBranch
from .tcn import TCN1d, TCN2d


def create_activation(name: str) -> nn.Module | None:
    if name == "none":
        return None
    if name == "relu":
        return nn.ReLU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    raise NotImplementedError(f"Activation function {name} is not implemented.")


__all__ = [
    "create_aggregator2d",
    "create_aggregator3d",
    "ConvBlock1d",
    "ConvBlock2d",
    "ConvBlock3d",
    "ConvModule1d",
    "ConvModule2d",
    "ConvModule3d",
    "IdenticalConvBlock1d",
    "IdenticalConvBlock2d",
    "IdenticalConvBlock3d",
    "IdenticalConvBlockConvParams",
    "PixelWiseConv1d",
    "PixelWiseConv2d",
    "PixelWiseConv3d",
    "HierarchicalConvDecoder1d",
    "HierarchicalConvDecoder2d",
    "HierarchicalConvDecoder3d",
    "HierarchicalConvEncoder1d",
    "HierarchicalConvEncoder2d",
    "HierarchicalConvEncoder3d",
    "ConvLSTM1d",
    "ConvLSTM2d",
    "GRU1d",
    "GRU2d",
    "ResNetBranch",
    "TCN1d",
    "TCN2d",
    "create_activation",
]
