from dataclasses import dataclass

from torch import Tensor

from .option import TransformOption


@dataclass
class MinMaxNormalizationOption(TransformOption):
    pass


class MinMaxNormalization:
    def __call__(self, x: Tensor) -> Tensor:
        return (x - x.min()) / (x.max() - x.min())
