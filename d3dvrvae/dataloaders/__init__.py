from torch.utils.data import DataLoader

from .datasets import create_dataset
from .transforms import TransformOption, create_transform
from .option import DataLoaderOption
from .basic import BasicDataLoaderOption, create_basic_dataloader


def create_dataloader(
    opt: DataLoaderOption,
    is_train: bool,
) -> tuple[DataLoader, DataLoader | None]:
    if isinstance(opt, BasicDataLoaderOption):
        return create_basic_dataloader(opt, is_train)
    raise NotImplementedError(f'{opt.__class__.__name__} is not implemented')
