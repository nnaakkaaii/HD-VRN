from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from omegaconf import MISSING
from torch import Tensor, int64, tensor
from torch.utils.data import Dataset

from ..transforms import Transform
from .ct import CT
from .option import DatasetOption


@dataclass
class SlicedCTDatasetOption(DatasetOption):
    root: Path = MISSING
    slice_index: list[int] = MISSING
    in_memory: bool = False
    content_phase: str = "all"
    motion_phase: str = "0"
    motion_aggregation: str = "concat"
    slice_axis: str = "y"  # y or z
    slice_num: int = MISSING


def create_sliced_ct_dataset(
    opt: SlicedCTDatasetOption, transform: Transform, is_train: bool
) -> Dataset:
    def slice_indexer(_: Tensor) -> Tensor:
        return tensor(opt.slice_index, dtype=int64)

    return SlicedCT(
        root=opt.root,
        slice_indexer=slice_indexer,
        transform=transform,
        in_memory=opt.in_memory,
        is_train=is_train,
        content_phase=opt.content_phase,
        motion_phase=opt.motion_phase,
        motion_aggregation=opt.motion_aggregation,
        slice_axis=opt.slice_axis,
        slice_num=opt.slice_num,
    )


class SlicedCT(CT):
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
        slice_axis: str = "y",
        slice_num: int = 0,
    ) -> None:
        super().__init__(
            root=root,
            slice_indexer=slice_indexer,
            transform=transform,
            in_memory=in_memory,
            is_train=is_train,
            content_phase=content_phase,
            motion_phase=motion_phase,
            motion_aggregation=motion_aggregation,
        )
        self.slice_axis = slice_axis
        self.slice_num = slice_num

    def __len__(self) -> int:
        return len(self.paths) * self.slice_num

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        output = super().__getitem__(index // self.slice_num)
        assert "xm" in output  # (n, _, d, h)
        assert "xm_0" in output  # (_, d, h)
        assert "xp" in output  # (n, _, d, h, w)
        assert "xp_0" in output  # (_, d, h, w)

        slice_index = index % self.slice_num
        if self.slice_axis == "y":
            # slice by h
            return {
                "xm": output["xm"][:, :, :, slice_index],
                "xm_0": output["xm_0"][:, :, slice_index],
                "xp": output["xp"][:, :, :, slice_index],
                "xp_0": output["xp_0"][:, :, slice_index],
            }
        elif self.slice_axis == "z":
            # slice by d
            return {
                "xm": output["xm"][:, :, slice_index],
                "xm_0": output["xm_0"][:, slice_index],
                "xp": output["xp"][:, :, slice_index],
                "xp_0": output["xp_0"][:, slice_index],
            }
        raise KeyError(f"unknown slice axis {self.slice_axis}")
