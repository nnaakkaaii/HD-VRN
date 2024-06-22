from dataclasses import dataclass, field

from torch import Tensor, cat, gather, int64, tensor, device
from torch.cuda import is_available
from torch.utils.data import Dataset
from torchvision import datasets

from ..transforms import Transform
from .option import DatasetOption
from .functions import optimize_output


@dataclass
class MovingMNISTDatasetOption(DatasetOption):
    root: str = "data"
    slice_index: list[int] = field(default_factory=lambda: [32])
    content_phase: str = "all"
    motion_phase: str = "0"
    motion_aggregation: str = "concat"


class MovingMNIST(datasets.MovingMNIST):
    PERIOD = 10

    def __init__(self, *args, **kwargs) -> None:
        self.slice_index = kwargs.pop("slice_index")
        self.content_phase = kwargs.pop("content_phase", "all")
        self.motion_phase = kwargs.pop("motion_phase", "0")
        self.motion_aggregator = kwargs.pop("motion_aggregation", "concat")
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        # (n, h, w)
        x_2d = self.data[idx].to(device("cuda:0") if is_available() else device("cpu"))
        if self.transform is not None:
            x_2d = self.transform(x_2d)
        x_2d = x_2d.squeeze(1)

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

        output = optimize_output(
            x_1d,
            x_1d_0,
            x_1d_t,
            x_1d_all,
            x_2d,
            x_2d_0,
            x_2d_t,
            x_2d_all,
            self.content_phase,
            self.motion_phase,
            self.motion_aggregator,
        )
        output["slice_idx"] = slice_idx
        output["idx_expanded"] = idx_expanded

        return output


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
        content_phase=opt.content_phase,
        motion_phase=opt.motion_phase,
        motion_aggregation=opt.motion_aggregation,
    )
