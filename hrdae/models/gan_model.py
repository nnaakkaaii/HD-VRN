from dataclasses import dataclass

from omegaconf import MISSING

from .losses import LossOption
from .networks import NetworkOption
from .optimizers import OptimizerOption
from .option import ModelOption
from .schedulers import SchedulerOption


@dataclass
class GANModelOption(ModelOption):
    generator: NetworkOption = MISSING
    discriminator: NetworkOption = MISSING
    optimizer_g: OptimizerOption = MISSING
    optimizer_d: OptimizerOption = MISSING
    scheduler: SchedulerOption = MISSING
    loss_g: dict[str, LossOption] = MISSING
    loss_g_coef: dict[str, float] = MISSING
    loss_d: dict[str, LossOption] = MISSING
    loss_d_coef: dict[str, float] = MISSING
