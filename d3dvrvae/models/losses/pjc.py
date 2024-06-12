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
        slice_idx: Tensor,
    ) -> Tensor:
        """
        :param input:
            1. (b, n, d, w, h)
            2. (b, n, d, w, h)
            3. (b, d, w, h)
            4. (b, d, w, h)
        :param target:
            1. (b, d, w, s)
            2. (b, d, w)
            3. (b, d, w, s)
            4. (b, d, w)
        :param slice_idx:
            1. (b, s)
            2. (b,)
            3. (b, s)
            4. (b,)
        :return:
        """
        if input.dim() == 4:
            input = input.unsqueeze(1)
        assert input.dim() == 5
        b, n, d, w, h = input.shape

        if target.dim() == 3 and slice_idx.dim() == 1:
            target = target.unsqueeze(3)
            slice_idx = slice_idx.unsqueeze(1)
        assert target.dim() == 4
        assert slice_idx.dim() == 2
        s = target.shape[3]

        # input: (b, n, d, w, h)
        # target: (b, d, w, s)
        # slice_idx: (b, s)
        assert input.shape == (b, n, d, w, h)
        assert target.shape == (b, d, w, s)
        assert slice_idx.shape == (b, s)

        # target: (b, n, d, w, s)
        target = target.unsqueeze(1).repeat(1, n, 1, 1, 1)
        assert target.shape == (b, n, d, w, s), f'{target.shape} != {(b, n, d, w, s)}'
        # slice_idx: (b, n, d, w, s)
        idx_expanded = slice_idx.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(b, n, d, w, s)
        assert idx_expanded.shape == (b, n, d, w, s)
        # input: (b, n, d, w, s)
        selected_slices = gather(input, -1, idx_expanded)
        assert selected_slices.shape == (b, n, d, w, s)

        return mse_loss(selected_slices, target)


if __name__ == '__main__':
    from torch import randn, randint

    def test():
        pjc_loss = PJCLoss()
        b, n, d, w, h, s = 32, 10, 50, 128, 128, 3
        # reconstructed_3d: (b, n, d, w, h)
        # input_2d: (b, d, w, s)
        # slice_idx: (b, s)
        reconstructed_3d = randn(b, n, d, w, h)
        input_2d = randn(b, d, w, s)
        slice_idx = randint(0, 127, (b, s))
        loss = pjc_loss(reconstructed_3d, input_2d, slice_idx)
        print(loss)
        # reconstructed_3d: (b, n, d, w, h)
        # input_2d: (b, d, w)
        # slice_idx: (b,)
        reconstructed_3d = randn(b, n, d, w, h)
        input_2d = randn(b, d, w)
        slice_idx = randint(0, 127, (b,))
        loss = pjc_loss(reconstructed_3d, input_2d, slice_idx)
        print(loss)
        # reconstructed_3d: (b, d, w, h)
        # input_2d: (b, d, w, s)
        # slice_idx: (b, s)
        reconstructed_3d = randn(b, d, w, h)
        input_2d = randn(b, d, w, s)
        slice_idx = randint(0, 127, (b, s))
        loss = pjc_loss(reconstructed_3d, input_2d, slice_idx)
        print(loss)
        # reconstructed_3d: (b, d, w, h)
        # input_2d: (b, d, w)
        # slice_idx: (b,)
        reconstructed_3d = randn(b, d, w, h)
        input_2d = randn(b, d, w)
        slice_idx = randint(0, 127, (b,))
        loss = pjc_loss(reconstructed_3d, input_2d, slice_idx)
        print(loss)

    test()
