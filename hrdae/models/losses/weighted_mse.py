from dataclasses import dataclass

from torch import Tensor, nn
from torch.nn.functional import mse_loss

from .option import LossOption


@dataclass
class WeightedMSELossOption(LossOption):
    weight_dynamic: float = 1.0


def create_weighted_mse_loss(opt: WeightedMSELossOption) -> nn.Module:
    return WeightedMSELoss(weight_dynamic=opt.weight_dynamic)


class WeightedMSELoss(nn.Module):
    def __init__(
        self,
        weight_dynamic: float,
    ) -> None:
        super().__init__()
        self.weight_dynamic = weight_dynamic

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert input.size() == target.size()

        loss_static = mse_loss(input, target)

        if input.size(1) == 1:
            return loss_static

        input_dynamic = input[:, 1:] - input[:, :-1]
        target_dynamic = target[:, 1:] - target[:, :-1]
        loss_dynamic = mse_loss(input_dynamic, target_dynamic)

        return loss_static + self.weight_dynamic * loss_dynamic
