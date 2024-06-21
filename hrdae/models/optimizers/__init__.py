from dataclasses import dataclass, field
from typing import Any, Iterable

from torch import Tensor
from torch.optim import Adam, Optimizer

from .option import OptimizerOption


@dataclass
class AdamOptimizerOption(OptimizerOption):
    lr: float = 0.0
    lrs: dict[str, float] = field(default_factory=dict)


def create_optimizer(
    opt: OptimizerOption,
    params: Iterable[Tensor] | Iterable[dict[str, Any]],
) -> Optimizer:
    if isinstance(opt, AdamOptimizerOption) and type(opt) is AdamOptimizerOption:
        if opt.lr > 0:
            return Adam([{"params": params, "lr": opt.lr}])
        if len(opt.lrs) > 0:
            return Adam([{"params": v, "lr": opt.lrs[k]} for k, v in params])
        raise ValueError("either lr or lrs must be set")
    raise NotImplementedError(f"{opt.__class__.__name__} is not implemented")
