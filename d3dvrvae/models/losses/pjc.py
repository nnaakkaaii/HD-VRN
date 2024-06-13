from dataclasses import dataclass

from torch import Tensor, gather, nn
from torch.nn.functional import mse_loss


@dataclass
class PJCLossOption:
    pass


def create_pjc_loss() -> nn.Module:
    return PJCLoss()


class PJCLoss(nn.Module):
    @property
    def required_kwargs(self) -> list[str]:
        return ["slice_idx"]

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        idx_expanded: Tensor,
    ) -> Tensor:
        """
        :param input: (b, n, d, h, w)
        :param target: (b, d, h, s)
        :param idx_expanded: (b, n, d, h, s)
        :return:
        """
        assert input.dim() == 5
        b, n, d, w, h = input.size()

        assert target.dim() == 4
        s = target.size(3)

        assert idx_expanded.dim() == 5

        # input: (b, n, d, h, w)
        # target: (b, d, h, s)
        # idx_expanded: (b, n, d, h, s)
        assert input.size() == (b, n, d, h, w)
        assert target.size() == (b, d, h, s)
        assert idx_expanded.size() == (b, n, d, h, s)

        # target: (b, n, d, h, s)
        target = target.unsqueeze(1).repeat(1, n, 1, 1, 1)
        assert target.size() == (b, n, d, h, s), f"{target.size()} != {(b, n, d, h, s)}"
        # input: (b, n, d, h, s)
        selected_slices = gather(input, -1, idx_expanded)
        assert selected_slices.size() == (b, n, d, h, s)

        return mse_loss(selected_slices, target)


if __name__ == "__main__":
    from torch import randint, randn

    def test():
        pjc_loss = PJCLoss()
        b, n, d, h, w, s = 32, 10, 50, 128, 128, 3
        # reconstructed_3d: (b, n, d, h, w)
        # input_2d: (b, d, h, s)
        # slice_idx: (b, n, d, h, s)
        reconstructed_3d = randn(b, n, d, h, w)
        input_2d = randn(b, d, h, s)
        slice_idx = (
            randint(0, 127, (b, s))
            .unsqueeze(1)
            .unsqueeze(2)
            .unsqueeze(3)
            .expand((b, n, d, h, s))
        )
        loss = pjc_loss(reconstructed_3d, input_2d, slice_idx)
        print(loss)

    test()
