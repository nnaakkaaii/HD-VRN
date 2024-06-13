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
        :param input: (b, n, c=1, d, h, w)
        :param target: (b, n, s, d, h)
        :param idx_expanded: (b, n, s, d, h)
        :return:
        """
        assert input.dim() == 6
        b, n, c, d, w, h = input.size()
        assert c == 1

        assert target.dim() == 5
        s = target.size(2)

        assert idx_expanded.dim() == 5

        # input: (b, n, c=1, d, h, w)
        # target: (b, n, s, d, h)
        # idx_expanded: (b, n, s, d, h)
        assert input.size() == (b, n, 1, d, h, w)
        assert target.size() == (b, n, s, d, h)
        assert idx_expanded.size() == (b, n, s, d, h)

        # input: (b, n, c=1, d, h, w) -> (b, n, d, h, w) -> (b, n, w, d, h)
        input = input.squeeze(2).permute(0, 1, 4, 2, 3)
        # input: (b, n, w, d, h) -> (b, n, s, d, h)
        selected_slices = gather(input, -1, idx_expanded)
        assert selected_slices.size() == (b, n, s, d, h)

        return mse_loss(selected_slices, target)


if __name__ == "__main__":
    from torch import randint, randn

    def test():
        pjc_loss = PJCLoss()
        b, n, c, d, h, w, s = 32, 10, 1, 50, 128, 128, 3
        # reconstructed_3d: (b, n, c, d, h, w)
        # input_2d: (b, n, s, d, h)
        # slice_idx: (b, n, s, d, h)
        reconstructed_3d = randn(b, n, c, d, h, w)
        input_2d = randn(b, n, s, d, h)
        slice_idx = (
            randint(0, 127, (b, s))
            .unsqueeze(1)
            .unsqueeze(3)
            .unsqueeze(4)
            .repeat(1, n, 1, d, h)
        )
        loss = pjc_loss(reconstructed_3d, input_2d, slice_idx)
        print(loss)

    test()
