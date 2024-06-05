from dataclasses import dataclass

from torch import nn

from .option import LossOption
from .pjc import PJCLossOption, create_pjc_loss


@dataclass
class MSELossOption(LossOption):
    pass


def create_loss(opt: LossOption) -> nn.Module:
    if isinstance(opt, MSELossOption):
        return nn.MSELoss()
    if isinstance(opt, PJCLossOption):
        return create_pjc_loss(opt)
    raise NotImplementedError(f"{opt.__class__.__name__} is not implemented")
