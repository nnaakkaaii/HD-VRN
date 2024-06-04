import torch

from .datasets import Data


def collate_fn(batch: list[tuple[Data, Data]]) -> tuple[Data, Data]:
    x = Data(torch.stack([item[0].data for item in batch]),
             torch.stack([item[0].latent for item in batch]) if batch[0][0].latent is not None else None)
    t = Data(torch.stack([item[1].data for item in batch]),
             torch.stack([item[1].latent for item in batch]) if batch[0][1].latent is not None else None)
    return x, t
