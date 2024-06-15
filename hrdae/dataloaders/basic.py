from dataclasses import dataclass

from omegaconf import MISSING
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from .datasets import DatasetOption, create_dataset
from .option import DataLoaderOption
from .transforms import TransformOption, create_transform


@dataclass
class BasicDataLoaderOption(DataLoaderOption):
    dataset: DatasetOption = MISSING
    transform: dict[str, TransformOption] = MISSING
    transform_order_train: list[str] = MISSING
    transform_order_val: list[str] = MISSING


def create_basic_dataloader(
    opt: BasicDataLoaderOption,
    is_train: bool,
) -> tuple[DataLoader, DataLoader | None]:
    transform_order = opt.transform_order_train if is_train else opt.transform_order_val
    transform = transforms.Compose(
        [create_transform(opt.transform[name]) for name in transform_order]
    )
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
            shuffle=is_train,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=opt.batch_size,
            shuffle=is_train,
        )
        return train_loader, val_loader

    return (
        DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=is_train,
        ),
        None,
    )
