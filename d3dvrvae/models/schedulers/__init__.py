from dataclasses import dataclass

from omegaconf import MISSING
from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR

from .option import SchedulerOption
from .typing import LRScheduler


@dataclass
class OneCycleLRSchedulerOption(SchedulerOption):
    max_lr: float = MISSING


def create_scheduler(
    opt: SchedulerOption,
    optimizer: Optimizer,
    n_epoch: int,
    steps_per_epoch: int,
) -> LRScheduler:
    if (
        isinstance(opt, OneCycleLRSchedulerOption)
        and type(opt) is OneCycleLRSchedulerOption
    ):
        return OneCycleLR(
            optimizer,
            max_lr=opt.max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=n_epoch,
        )
    raise NotImplementedError(f"{opt.__class__.__name__} is not implemented")
