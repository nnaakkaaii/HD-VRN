from dataclasses import dataclass

from torch.utils.data import DataLoader, Dataset


@dataclass
class DataLoaderOption:
    batch_size: int = 32


def create_dataloader(
    opt: DataLoaderOption,
    dataset: Dataset,
    is_train: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=is_train,
    )
