from dataclasses import dataclass
from typing import Callable

from omegaconf import MISSING
from torch import cat, Tensor
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

    aggregate_dim: int = -1


def get_aggregating_collate_fn(
    aggregate_dim: int,
) -> Callable[[list[dict[str, Tensor]]], dict[str, Tensor]]:
    def collate_fn(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        old: dict[str, list[Tensor]] = {}
        for item in batch:
            for k_item, v_item in item.items():
                if k_item not in old:
                    old[k_item] = []
                old[k_item].append(v_item)
        new: dict[str, Tensor] = {}
        for k_old, v_old in old.items():
            new[k_old] = cat(v_old, dim=aggregate_dim)
        return new

    return collate_fn


def create_basic_dataloader(
    opt: BasicDataLoaderOption,
    is_train: bool,
) -> tuple[DataLoader, DataLoader | None]:
    transform_order = opt.transform_order_train if is_train else opt.transform_order_val
    transform = transforms.Compose(
        [create_transform(opt.transform[name]) for name in transform_order]
    )
    dataset = create_dataset(opt.dataset, transform, is_train)
    collate_fn = None
    if opt.aggregate_dim >= 0:
        collate_fn = get_aggregating_collate_fn(opt.aggregate_dim)

    if is_train:
        train_size = int(opt.train_val_ratio * len(dataset))  # type: ignore
        val_size = len(dataset) - train_size  # type: ignore
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=is_train,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=opt.batch_size,
            shuffle=is_train,
            collate_fn=collate_fn,
        )
        return train_loader, val_loader

    return (
        DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=is_train,
            collate_fn=collate_fn,
        ),
        None,
    )
