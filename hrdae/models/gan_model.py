from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import MISSING
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from .functions import save_reconstructed_images
from .losses import LossOption, create_loss
from .networks import NetworkOption, create_network
from .optimizers import OptimizerOption, create_optimizer
from .option import ModelOption
from .schedulers import LRScheduler, SchedulerOption, create_scheduler
from .typing import Model


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
