from torch import nn

from .option import LossOption


class MSELossOption(LossOption):
    pass


def create_loss(opt: LossOption) -> nn.Module:
    if isinstance(opt, MSELossOption):
        return nn.MSELoss()
    raise NotImplementedError(f'{opt.__class__.__name__} is not implemented')
