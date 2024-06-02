from dataclasses import dataclass
from typing import Any, Iterable

from torch import Tensor
from torch.optim import Optimizer, Adam
from omegaconf import MISSING

from .option import OptimizerOption


@dataclass
class AdamOptimizerOption(OptimizerOption):
    lr: float = MISSING


def create_optimizer(opt: OptimizerOption,
                     params: Iterable[Tensor] | Iterable[dict[str, Any]],
                     ) -> Optimizer:
    if isinstance(opt, AdamOptimizerOption):
        return Adam(params, lr=opt.lr)
    raise NotImplementedError(f'{opt.__class__.__name__} is not implemented')
