from dataclasses import dataclass

from torch.utils.data import Dataset
from torchvision import datasets

from .option import DatasetOption
from ..transforms import Transform


@dataclass
class MNISTDatasetOption(DatasetOption):
    root: str = 'data'


def create_mnist_dataset(opt: MNISTDatasetOption,
                         transform: Transform,
                         is_train: bool,
                         ) -> Dataset:
    return datasets.MNIST(opt.root,
                          train=is_train,
                          download=True,
                          transform=transform,
                          )
