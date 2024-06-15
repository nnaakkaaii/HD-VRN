from dataclasses import dataclass, field

from torch import Tensor, cat, gather, int64, tensor
from torch.utils.data import Dataset
from torchvision import datasets

from ..transforms import Transform
from .option import DatasetOption


@dataclass
class MovingMNISTDatasetOption(DatasetOption):
    root: str = "data"
    slice_index: list[int] = field(default_factory=lambda: [32])


class MovingMNIST(datasets.MovingMNIST):
    PERIOD = 10

    def __init__(self, *args, **kwargs) -> None:
        self.slice_index = kwargs.pop("slice_index")
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

        # (n, h, s) -> (n, s, h)
        x_1d = x_1d.permute(0, 2, 1)
        # (h, s) -> (s, h)
        x_1d_0 = x_1d_0.permute(1, 0)
        x_1d_t = x_1d_t.permute(1, 0)
        # (2 * s, h)
        x_1d_all = cat([x_1d_0, x_1d_t], dim=0)
        # (n, h, w) -> (n, c, h, w)
        x_2d = x_2d.unsqueeze(1)
        # (h, w) -> (c, h, w)
        x_2d_0 = x_2d_0.unsqueeze(0)
        x_2d_t = x_2d_t.unsqueeze(0)
        # (2 * c, h, w)
        x_2d_all = cat([x_2d_0, x_2d_t], dim=0)
        # (n, h, s) -> (n, s, h)
        idx_expanded = idx_expanded.permute(0, 2, 1)

        return {
            "x-": x_1d,  # (n, s, h)
            "x-_0": x_1d_0,  # (s, h)
            "x-_t": x_1d_t,  # (s, h)
            "x-_all": x_1d_all,  # (2 * s, h)
            "x+": x_2d,  # (n, c, h, w)
            "x+_0": x_2d_0,  # (c, h, w)
            "x+_t": x_2d_t,  # (c, h, w)
            "x+_all": x_2d_all,  # (2 * c, h, w)
            "slice_idx": slice_idx,  # (s,)
            "idx_expanded": idx_expanded,  # (n, s, h)
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


if __name__ == "__main__":

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
