from dataclasses import dataclass

from omegaconf import MISSING
from torch.utils.data import DataLoader, random_split

from .option import DataLoaderOption
from .datasets import DatasetOption, create_dataset
from .transforms import TransformOption, create_transform
from .functions import collate_fn


@dataclass
class BasicDataLoaderOption(DataLoaderOption):
    dataset: DatasetOption = MISSING
    transform: TransformOption = MISSING


def create_basic_dataloader(
        opt: BasicDataLoaderOption,
        is_train: bool,
) -> tuple[DataLoader, DataLoader | None]:
    transform = create_transform(opt.transform)
    dataset = create_dataset(opt.dataset, transform, is_train)

    if is_train:
        train_size = int(opt.train_val_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            collate_fn=collate_fn,
            shuffle=is_train,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=opt.batch_size,
            collate_fn=collate_fn,
            shuffle=is_train,
        )
        return train_loader, val_loader

    return DataLoader(
        dataset,
        batch_size=opt.batch_size,
        collate_fn=collate_fn,
        shuffle=is_train,
    ), None
