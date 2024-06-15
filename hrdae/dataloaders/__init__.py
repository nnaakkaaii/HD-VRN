from torch.utils.data import DataLoader

from .basic import BasicDataLoaderOption, create_basic_dataloader
from .datasets import create_dataset
from .option import DataLoaderOption
from .transforms import TransformOption, create_transform


def create_dataloader(
    opt: DataLoaderOption,
    is_train: bool,
) -> tuple[DataLoader, DataLoader | None]:
    if isinstance(opt, BasicDataLoaderOption) and type(opt) is BasicDataLoaderOption:
        return create_basic_dataloader(opt, is_train)
    raise NotImplementedError(f"{opt.__class__.__name__} is not implemented")


__all__ = [
    "create_dataloader",
    "create_dataset",
    "create_transform",
    "TransformOption",
]
