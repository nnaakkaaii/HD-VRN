from dataclasses import dataclass, field

from torch import Tensor, tensor, int64, gather
from torch.utils.data import Dataset
from torchvision import datasets

from ..transforms import Transform
from .option import DatasetOption


@dataclass
class MovingMNISTDatasetOption(DatasetOption):
    root: str = "data"
    slice_index: list[int] = field(default_factory=lambda: [16, 32, 48])


class MovingMNIST(datasets.MovingMNIST):
    PERIOD = 10

    def __init__(self, *args, **kwargs) -> None:
        self.slice_index = kwargs.pop('slice_index')
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        # (n, h, w)
        x_2d = super().__getitem__(idx).squeeze(1)
        n, h, w = x_2d.size()

        # (s,)
        slice_idx = tensor(self.slice_index, dtype=int64)
        # (n, h, s)
        idx_expanded = slice_idx.unsqueeze(0).unsqueeze(1).repeat(n, h, 1)
        # (n, h, s)
        x_1d = gather(x_2d, -1, idx_expanded)
        # (h, s)
        x_1d_0 = x_1d[0]
        x_1d_t = x_1d[self.PERIOD // 2]
        # (h, w)
        x_2d_0 = x_2d[0]
        x_2d_t = x_2d[self.PERIOD // 2]

        # (n, h, s) -> (b, n, h, s) -> (b, n, s, h)
        x_1d = x_1d.unsqueeze(0).permute(0, 1, 3, 2)
        # (h, s) -> (b, h, s) -> (b, s, h)
        x_1d_0 = x_1d_0.unsqueeze(0).permute(0, 2, 1)
        x_1d_t = x_1d_t.unsqueeze(0).permute(0, 2, 1)
        # (n, h, w) -> (b, n, c, h, w)
        x_2d = x_2d.unsqueeze(0).unsqueeze(2)
        # (h, w) -> (b, c, h, w)
        x_2d_0 = x_2d_0.unsqueeze(0).unsqueeze(1)
        x_2d_t = x_2d_t.unsqueeze(0).unsqueeze(1)
        # (s,) -> (b, s)
        slice_idx = slice_idx.unsqueeze(0)
        # (n, h, s) -> (b, n, h, s) -> (b, n, s, h)
        idx_expanded = idx_expanded.unsqueeze(0).permute(0, 1, 3, 2)

        return {
            "x_1d": x_1d,  # (b, n, s, h)
            "x_1d_0": x_1d_0,  # (b, s, h)
            "x_1d_t": x_1d_t,  # (b, s, h)
            "x_2d": x_2d,  # (b, n, c, h, w)
            "x_2d_0": x_2d_0,  # (b, c, h, w)
            "x_2d_t": x_2d_t,  # (b, c, h, w)
            "slice_idx": slice_idx,  # (b, s)
            "idx_expanded": idx_expanded,  # (b, n, s, h)
        }


def create_moving_mnist_dataset(
    opt: MovingMNISTDatasetOption,
    transform: Transform,
    is_train: bool,
) -> Dataset:
    return MovingMNIST(
        opt.root,
        slice_index=opt.slice_index,
        split="train" if is_train else "test",
        download=True,
        transform=transform,
    )


if __name__ == '__main__':
    def test():
        from torchvision import transforms

        option = MovingMNISTDatasetOption(
            root="data",
        )
        dataset = create_moving_mnist_dataset(
            option,
            transform=transforms.Compose([]),
            is_train=True,
        )
        data = dataset[0]
        for k, v in data.items():
            print(k, v.shape)

    test()
