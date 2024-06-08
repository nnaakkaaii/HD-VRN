from dataclasses import dataclass

from torch import Tensor

from .typing import Transform


@dataclass
class MinMaxNormalizationOption:
    pass


class MinMaxNormalization(Transform):
    def __call__(self, x: Tensor) -> Tensor:
        return (x - x.min()) / (x.max() - x.min())
