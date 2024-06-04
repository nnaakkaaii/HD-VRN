from dataclasses import dataclass
from typing import Any, Iterable

from omegaconf import MISSING
from torch import Tensor
from torch.optim import Adam, Optimizer

from .option import OptimizerOption


@dataclass
class AdamOptimizerOption(OptimizerOption):
    lr: float = MISSING


def create_optimizer(
    opt: OptimizerOption,
    params: Iterable[Tensor] | Iterable[dict[str, Any]],
) -> Optimizer:
    if isinstance(opt, AdamOptimizerOption):
        return Adam(params, lr=opt.lr)
    raise NotImplementedError(f"{opt.__class__.__name__} is not implemented")
