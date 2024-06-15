from dataclasses import dataclass

from torch import Tensor, gather, nn
from torch.nn.functional import mse_loss


@dataclass
class PJC2dLossOption:
    pass


@dataclass
class PJC3dLossOption:
    pass


def create_pjc2d_loss() -> nn.Module:
    return PJC2dLoss()


def create_pjc3d_loss() -> nn.Module:
    return PJC3dLoss()


class PJC2dLoss(nn.Module):
    @property
    def required_kwargs(self) -> list[str]:
        return ["idx_expanded"]

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
        assert idx_expanded.dim() == 4
        s = idx_expanded.size(2)

        # input: (b, n, c=1, h, w)
        # target: (b, n, c=1, h, w)
        # idx_expanded: (b, n, s, h)
        assert input.size() == (b, n, 1, h, w)
        assert target.size() == (b, n, 1, h, w)
        assert idx_expanded.size() == (b, n, s, h)

        # (b, n, c=1, h, w) -> (b, n, h, w) -> (b, n, w, h)
        input = input.squeeze(2).permute(0, 1, 3, 2)
        target = target.squeeze(2).permute(0, 1, 3, 2)
        # (b, n, w, h) -> (b, n, s, h)
        input_slices = gather(input, -1, idx_expanded)
        target_slices = gather(target, -1, idx_expanded)
        assert input_slices.size() == (b, n, s, h)
        assert target_slices.size() == (b, n, s, h)

        return mse_loss(input_slices, target_slices)


class PJC3dLoss(nn.Module):
    @property
    def required_kwargs(self) -> list[str]:
        return ["idx_expanded"]

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        idx_expanded: Tensor,
    ) -> Tensor:
        """
        :param input: (b, n, c=1, d, h, w)
        :param target: (b, n, c=1, d, h, w)
        :param idx_expanded: (b, n, s, d, h)
        :return:
        """
        assert input.dim() == 6
        b, n, c, d, h, w = input.size()
        assert c == 1
        assert idx_expanded.dim() == 5
        s = idx_expanded.size(2)

        # input: (b, n, c=1, d, h, w)
        # target: (b, n, c=1, d, h, w)
        # idx_expanded: (b, n, s, d, h)
        assert input.size() == (b, n, 1, d, h, w)
        assert target.size() == (b, n, 1, d, h, w)
        assert idx_expanded.size() == (b, n, s, d, h)

        # (b, n, c=1, d, h, w) -> (b, n, d, h, w) -> (b, n, w, d, h)
        input = input.squeeze(2).permute(0, 1, 4, 2, 3)
        target = target.squeeze(2).permute(0, 1, 4, 2, 3)
        # (b, n, w, d, h) -> (b, n, s, d, h)
        input_slices = gather(input, -1, idx_expanded)
        target_slices = gather(target, -1, idx_expanded)
        assert input_slices.size() == (b, n, s, d, h)
        assert target_slices.size() == (b, n, s, d, h)

        return mse_loss(input_slices, target_slices)


if __name__ == "__main__":
    from torch import randint, randn

    def test():
        b, n, c, d, h, w, s = 32, 10, 1, 50, 128, 128, 3

        pjc2d_loss = PJC2dLoss()
        # reconstructed_2d: (b, n, c, h, w)
        # input_2d: (b, n, c, h, w)
        # slice_idx: (b, n, s, h)
        reconstructed_2d = randn(b, n, c, h, w)
        input_2d = randn(b, n, c, h, w)
        slice_idx = randint(0, 127, (b, s)).unsqueeze(1).unsqueeze(3).repeat(1, n, 1, h)
        loss = pjc2d_loss(reconstructed_2d, input_2d, slice_idx)
        print(loss)

        pjc3d_loss = PJC3dLoss()
        # reconstructed_3d: (b, n, c, d, h, w)
        # input_3d: (b, n, c, d, h, w)
        # slice_idx: (b, n, s, d, h)
        reconstructed_3d = randn(b, n, c, d, h, w)
        input_3d = randn(b, n, c, d, h, w)
        slice_idx = (
            randint(0, 127, (b, s))
            .unsqueeze(1)
            .unsqueeze(3)
            .unsqueeze(4)
            .repeat(1, n, 1, d, h)
        )
        loss = pjc3d_loss(reconstructed_3d, input_3d, slice_idx)
        print(loss)

    test()
