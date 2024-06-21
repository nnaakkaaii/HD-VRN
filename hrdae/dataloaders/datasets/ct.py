import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from omegaconf import MISSING
from torch import Tensor, cat, from_numpy, gather, int64, tensor, where
from torch.utils.data import Dataset
from tqdm import tqdm

from ..transforms import Transform
from .option import DatasetOption
from .functions import optimize_output


@dataclass
class CTDatasetOption(DatasetOption):
    root: Path = MISSING
    slice_index: list[int] = MISSING
    threshold: float = 0.1
    min_occupancy: float = 0.2
    in_memory: bool = False
    content_phase: str = "all"
    motion_phase: str = "0"
    motion_aggregation: str = "concat"


class BasicSliceIndexer:
    def __init__(self, threshold: float = 0.1, min_occupancy: float = 0.2) -> None:
        self.threshold = threshold
        self.min_occupancy = min_occupancy

    def __call__(self, x: Tensor) -> Tensor:
        mask = where(x > self.threshold, 1, 0)
        choices = []
        n, d, h, w = x.size()
        for i in range(w):
            if mask[:, :, :, i].sum() < self.min_occupancy * n * d * h:
                continue
            choices.append(i)

        if len(choices) > 0:
            return tensor([random.choice(choices)], dtype=int64)

        return tensor([int(mask.sum(dim=(0, 1)).argmax())], dtype=int64)


def create_ct_dataset(
    opt: CTDatasetOption, transform: Transform, is_train: bool
) -> Dataset:
    slice_indexer: Callable[[Tensor], Tensor]
    if len(opt.slice_index) == 0:
        slice_indexer = BasicSliceIndexer(opt.threshold, opt.min_occupancy)
    else:

        def slice_indexer(_: Tensor) -> Tensor:
            return tensor(opt.slice_index, dtype=int64)

    return CT(
        root=opt.root,
        slice_indexer=slice_indexer,
        transform=transform,
        in_memory=opt.in_memory,
        is_train=is_train,
        content_phase=opt.content_phase,
        motion_phase=opt.motion_phase,
        motion_aggregation=opt.motion_aggregation,
    )


class CT(Dataset):
    TRAIN_PER_TEST = 4
    PERIOD = 10

    def __init__(
        self,
        root: Path,
        slice_indexer: Callable[[Tensor], Tensor],
        transform: Transform | None = None,
        in_memory: bool = True,
        is_train: bool = True,
        content_phase: str = "all",
        motion_phase: str = "0",
        motion_aggregation: str = "concat",  # "concat" | "sum"
    ) -> None:
        super().__init__()

        self.paths = []
        data_root = root / self.__class__.__name__
        for i, path in enumerate(sorted(data_root.glob("**/*"))):
            if is_train and i % (1 + self.TRAIN_PER_TEST) != 0:
                self.paths.append(path)
            elif not is_train and i % (1 + self.TRAIN_PER_TEST) == 0:
                self.paths.append(path)

        self.data: list[Tensor] = []
        if in_memory:
            for path in tqdm(self.paths, desc="loading datasets..."):
                t = from_numpy(np.load(path)["arr_0"])
                if transform is not None:
                    t = transform(t)
                self.data.append(t)

        self.slice_indexer = slice_indexer
        self.transform = transform
        self.in_memory = in_memory
        self.content_phase = content_phase
        self.motion_phase = motion_phase
        self.motion_aggregation = motion_aggregation

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if len(self.data) > 0:
            assert self.in_memory
            x_3d = self.data[index]
        else:  # not in memory
            assert not self.in_memory
            x_3d = from_numpy(np.load(str(self.paths[index]))["arr_0"])
            if self.transform is not None:
                x_3d = self.transform(x_3d)

        x_3d = x_3d.float()
        n, d, h, w = x_3d.size()
        assert n == self.PERIOD, f"expected {self.PERIOD} but got {n}"

        # (s,)
        slice_idx = self.slice_indexer(x_3d)
        # (n, d, h, s)
        idx_expanded = (
            slice_idx.unsqueeze(0).unsqueeze(1).unsqueeze(2).repeat(n, d, h, 1)
        )

        # (n, d, h, s)
        x_2d = gather(x_3d, -1, idx_expanded)
        # (d, h, s)
        x_2d_0 = x_2d[0]
        x_2d_t = x_2d[self.PERIOD // 2]
        # (d, h, w)
        x_3d_0 = x_3d[0]
        x_3d_t = x_3d[self.PERIOD // 2]

        # (n, d, h, s) -> (n, s, d, h)
        x_2d = x_2d.permute(0, 3, 1, 2)
        # (d, h, s) -> (s, d, h)
        x_2d_0 = x_2d_0.permute(2, 0, 1)
        x_2d_t = x_2d_t.permute(2, 0, 1)
        # (2 * s, d, h)
        x_2d_all = cat([x_2d_0, x_2d_t], dim=0)
        # (n, d, h, w) -> (n, c, d, h, w)
        x_3d = x_3d.unsqueeze(1)
        # (d, h, w) -> (c, d, h, w)
        x_3d_0 = x_3d_0.unsqueeze(0)
        x_3d_t = x_3d_t.unsqueeze(0)
        # (2 * c, d, h, w)
        x_3d_all = cat([x_3d_0, x_3d_t], dim=0)
        # (n, d, h, s) -> (n, s, d, h)
        idx_expanded = idx_expanded.permute(0, 3, 1, 2)

        output = optimize_output(
            x_2d,
            x_2d_0,
            x_2d_t,
            x_2d_all,
            x_3d,
            x_3d_0,
            x_3d_t,
            x_3d_all,
            self.content_phase,
            self.motion_phase,
            self.motion_aggregation,
        )
        output["slice_idx"] = slice_idx
        output["idx_expanded"] = idx_expanded

        return output
