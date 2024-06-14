from dataclasses import dataclass

from torch import Tensor, gather, nn
from torch.nn.functional import mse_loss


@dataclass
class PJCLoss2dOption:
    pass


@dataclass
class PJCLoss3dOption:
    pass


def create_pjc_loss2d() -> nn.Module:
    return PJCLoss2d()


def create_pjc_loss3d() -> nn.Module:
    return PJCLoss3d()


class PJCLoss2d(nn.Module):
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
        :param input: (b, n, c=1, h, w)
        :param target: (b, n, s, h)
        :param idx_expanded: (b, n, s, h)
        :return:
        """
        assert input.dim() == 5
        b, n, c, w, h = input.size()
        assert c == 1

        assert target.dim() == 4
        s = target.size(2)

        assert idx_expanded.dim() == 4

        # input: (b, n, c=1, h, w)
        # target: (b, n, s, h)
        # idx_expanded: (b, n, s, h)
        assert input.size() == (b, n, 1, h, w)
        assert target.size() == (b, n, s, h)
        assert idx_expanded.size() == (b, n, s, h)

        # input: (b, n, c=1, h, w) -> (b, n, h, w) -> (b, n, w, h)
        input = input.squeeze(2).permute(0, 1, 3, 2)
        # input: (b, n, w, h) -> (b, n, s, h)
        selected_slices = gather(input, -1, idx_expanded)
        assert selected_slices.size() == (b, n, s, h)

        return mse_loss(selected_slices, target)


class PJCLoss3d(nn.Module):
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
        b, n, c, d, h, w = input.size()
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
        b, n, c, d, h, w, s = 32, 10, 1, 50, 128, 128, 3

        pjc_loss2d = PJCLoss2d()
        # reconstructed_2d: (b, n, c, h, w)
        # input_2d: (b, n, s, h)
        # slice_idx: (b, n, s, h)
        reconstructed_2d = randn(b, n, c, h, w)
        input_1d = randn(b, n, s, h)
        slice_idx = (
            randint(0, 127, (b, s))
            .unsqueeze(1)
            .unsqueeze(3)
            .repeat(1, n, 1, h)
        )
        loss = pjc_loss2d(reconstructed_2d, input_1d, slice_idx)
        print(loss)

        pjc_loss3d = PJCLoss3d()
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
        loss = pjc_loss3d(reconstructed_3d, input_2d, slice_idx)
        print(loss)

    test()
