from dataclasses import dataclass

from torch import Tensor, device
from torch.cuda import is_available

from .option import TransformOption


@dataclass
class MinMaxNormalizationOption(TransformOption):
    pass


class MinMaxNormalization:
    def __call__(self, x: Tensor) -> Tensor:
        x = x.to(device("cuda:0") if is_available() else device("cpu"))
        x = (x - x.min()) / (x.max() - x.min()).cpu()
        return x
