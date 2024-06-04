from abc import ABCMeta, abstractmethod

from torch.utils.data import DataLoader


class Model(metaclass=ABCMeta):
    @abstractmethod
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              n_epoch: int,
              result_dir: str,
              ) -> None:
        pass
