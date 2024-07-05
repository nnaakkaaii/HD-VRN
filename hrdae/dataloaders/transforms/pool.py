from dataclasses import dataclass, field

from torch import Tensor
from torch.nn.functional import avg_pool2d, avg_pool3d

from .option import TransformOption
from .typing import Transform


@dataclass
class Pool2dOption(TransformOption):
    pool_size: list[int] = field(default_factory=lambda: [4, 4])


def create_pool2d(opt: Pool2dOption) -> Transform:
    return Pool2d(opt.pool_size)


class Pool2d:
    def __init__(self, pool_size: list[int]):
        self.pool_size = pool_size

    def __call__(self, x: Tensor) -> Tensor:
        return avg_pool2d(x, self.pool_size)


@dataclass
class Pool3dOption(TransformOption):
    pool_size: list[int] = field(default_factory=lambda: [1, 4, 4])


def create_pool3d(opt: Pool3dOption) -> Transform:
    return Pool3d(opt.pool_size)


class Pool3d:
    def __init__(self, pool_size: list[int]):
        self.pool_size = pool_size

    def __call__(self, x: Tensor) -> Tensor:
        return avg_pool3d(x, self.pool_size)
