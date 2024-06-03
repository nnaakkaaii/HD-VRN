from dataclasses import dataclass

from torch import nn, Tensor

from .option import LossOption
from ..datasets import Data


@dataclass
class MSELossOption(LossOption):
    eval_latent: bool = False


class MSELoss(nn.MSELoss):
    def __init__(self, *args, **kwargs) -> None:
        self.eval_latent = kwargs.pop('eval_latent')
        super().__init__(*args, **kwargs)

    def forward(self, t: Data, y: Data) -> Tensor:
        if self.eval_latent:
            return super().forward(t.latent, y.latent)
        return super().forward(t.data, y.data)


def create_loss(opt: LossOption) -> nn.Module:
    if isinstance(opt, MSELossOption):
        return MSELoss(eval_latent=opt.eval_latent)
    raise NotImplementedError(f"{opt.__class__.__name__} is not implemented")
