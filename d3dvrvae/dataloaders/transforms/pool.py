from dataclasses import dataclass, field

from torch import Tensor
from torch.nn.functional import avg_pool3d

from .typing import Transform


@dataclass
class Pool3dOption(Transform):
    pool_size: list[int] = field(default_factory=lambda: [1, 4, 4])


class Pool3d(Transform):
    def __init__(self, pool_size: list[int]):
        self.pool_size = pool_size

    def __call__(self, x: Tensor) -> Tensor:
        return avg_pool3d(x, self.pool_size)
