from abc import ABCMeta, abstractmethod
from pathlib import Path

from torch.utils.data import DataLoader


class Model(metaclass=ABCMeta):
    @abstractmethod
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epoch: int,
        result_dir: Path,
        debug: bool,
    ) -> float:
        pass
