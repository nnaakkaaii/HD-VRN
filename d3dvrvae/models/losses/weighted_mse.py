from dataclasses import dataclass

from torch import Tensor, nn
from torch.nn.functional import mse_loss


@dataclass
class WeightedMSELossOption:
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
        b, n, d, w, h = input.size()
        assert target.size() == (b, n, d, w, h)

        loss_static = mse_loss(input, target)

        if n == 1:
            return loss_static

        input_dynamic = input[:, 1:] - input[:, :-1]
        target_dynamic = target[:, 1:] - target[:, :-1]
        loss_dynamic = mse_loss(input_dynamic, target_dynamic)

        return loss_static + self.weight_dynamic * loss_dynamic


if __name__ == "__main__":
    from torch import randn

    def test():
        wmse = WeightedMSELoss(weight_dynamic=0.5)
        b, n, d, w, h = 32, 10, 50, 128, 128

        input = randn(b, n, d, w, h)
        target = randn(b, n, d, w, h)

        loss = wmse(input, target)
        print(loss)

    test()
