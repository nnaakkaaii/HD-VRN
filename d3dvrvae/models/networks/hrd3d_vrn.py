# Licensed under the Apache License, Version 2.0 (the "License");
# Hierarchical Recurrent Disentangled 3D Video Reconstruction Network (HRD3D-VRN)

from dataclasses import dataclass, field

from torch import nn

from .option import NetworkOption
from .modules import ConvHierarchicalEncoder3d


@dataclass
class HRD3DVRNOption(NetworkOption):
    pass


def create_hrd3dvrn(opt: HRD3DVRNOption) -> nn.Module:
    pass


ContentEncoder3d = ConvHierarchicalEncoder3d


class HRD3DVRN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        pass
