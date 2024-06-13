import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from omegaconf import MISSING
from torch import Tensor, from_numpy, int32, tensor, where
from torch.utils.data import Dataset
from tqdm import tqdm

from ..transforms import Transform


@dataclass
class CTDatasetOption:
    root: Path = MISSING
    threshold: float = 0.1
    min_occupancy: float = 0.2
    in_memory: bool = False


class BasicSliceIndexer:
    def __init__(self, threshold: float = 0.1, min_occupancy: float = 0.2) -> None:
        self.threshold = threshold
        self.min_occupancy = min_occupancy

    def __call__(self, x: Tensor) -> int:
        mask = where(x > self.threshold, 1, 0)
        choices = []
        n, d, h, w = x.size()
        for i in range(w):
            if mask[:, :, :, i].sum() < self.min_occupancy * n * d * h:
                continue
            choices.append(i)

        if len(choices) > 0:
            return random.choice(choices)

        return int(mask.sum(dim=(0, 1)).argmax())


def create_ct_dataset(
    opt: CTDatasetOption, transform: Transform, is_train: bool
) -> Dataset:
    return CT(
        root=opt.root,
        slice_indexer=BasicSliceIndexer(opt.threshold, opt.min_occupancy),
        transform=transform,
        in_memory=opt.in_memory,
        is_train=is_train,
    )


class CT(Dataset):
    TRAIN_PER_TEST = 4
    PERIOD = 10

    def __init__(
        self,
        root: Path,
        slice_indexer: Callable[[Tensor], int],
        transform: Transform | None = None,
        in_memory: bool = True,
        is_train: bool = True,
    ) -> None:
        super().__init__()

        self.paths = []
        for i, path in enumerate(sorted(root.glob("**/*"))):
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

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        if len(self.data) > 0:
            assert self.in_memory
            t = self.data[index]
        else:  # not in memory
            assert not self.in_memory
            t = from_numpy(np.load(str(self.paths[index]))["arr_0"])
            if self.transform is not None:
                t = self.transform(t)

        n = t.size(0)
        assert n == self.PERIOD, f"expected {self.PERIOD} but got {n}"

        slice_idx = self.slice_indexer(t)

        # (n, d, h, w) -> (b, n, d, h, w)
        t = t.unsqueeze(0)

        x = t[:, :, :, :, slice_idx]
        x_0 = t[:, 0, :, :, :]
        x_T = t[:, self.PERIOD // 2, :, :, :]

        return {
            "x": x,  # (b, n, d, h)
            "x_0": x_0,  # (b, d, h, w)
            "x_T": x_T,  # (b, d, h, w)
            "t": t,  # (b, n, d, h, w)
            "slice_idx": tensor(slice_idx, dtype=int32),
        }
