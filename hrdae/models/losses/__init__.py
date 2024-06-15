from dataclasses import dataclass

from torch import Tensor, float32, nn, tensor

from .option import LossOption
from .pjc import PJC2dLossOption, PJC3dLossOption, create_pjc2d_loss, create_pjc3d_loss
from .weighted_mse import WeightedMSELossOption, create_weighted_mse_loss


@dataclass
class MSELossOption(LossOption):
    pass


def create_loss(opt: LossOption) -> nn.Module:
    if isinstance(opt, MSELossOption) and type(opt) is MSELossOption:
        return nn.MSELoss()
    if isinstance(opt, WeightedMSELossOption) and type(opt) is WeightedMSELossOption:
        return create_weighted_mse_loss(opt)
    if isinstance(opt, PJC2dLossOption) and type(opt) is PJC2dLossOption:
        return create_pjc2d_loss()
    if isinstance(opt, PJC3dLossOption) and type(opt) is PJC3dLossOption:
        return create_pjc3d_loss()
    raise NotImplementedError(f"{opt.__class__.__name__} is not implemented")


class LossMixer(nn.Module):
    def __init__(self, loss: dict[str, nn.Module], loss_coef: dict[str, float]):
        super().__init__()

        keys = sorted(list(loss.keys()))
        self.loss = nn.ModuleList([loss[key] for key in keys])
        self.loss_coef = [loss_coef[key] for key in keys]

    def forward(self, input: Tensor, target: Tensor, **kwargs) -> Tensor:
        loss = tensor(0, device=input.device, dtype=float32)
        for f, coef in zip(self.loss, self.loss_coef):
            kw = {}
            if hasattr(f, "required_kwargs"):
                kw |= {k: kwargs[k] for k in f.required_kwargs}
            loss += coef * f(input, target, **kw)
        return loss
