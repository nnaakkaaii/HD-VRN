from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

from ..datasets import Data


@dataclass
class DataLoaderOption:
    batch_size: int = 32


def collate_fn(batch: list[tuple[Data, Data]]) -> tuple[Data, Data]:
    x = Data(torch.stack([item[0].data for item in batch]),
             torch.stack([item[0].latent for item in batch]) if batch[0][0].latent is not None else None)
    t = Data(torch.stack([item[1].data for item in batch]),
             torch.stack([item[1].latent for item in batch]) if batch[0][1].latent is not None else None)
    return x, t


def create_dataloader(
    opt: DataLoaderOption,
    dataset: Dataset,
    is_train: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=opt.batch_size,
        collate_fn=collate_fn,
        shuffle=is_train,
    )
