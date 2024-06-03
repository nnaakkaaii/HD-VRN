from dataclasses import dataclass

from torch.utils.data import Dataset
from torchvision import datasets

from ..transforms import Transform
from .typing import Data
from .option import DatasetOption


@dataclass
class MNISTDatasetOption(DatasetOption):
    root: str = "data"
    input_as_label: bool = True


class MNIST(datasets.MNIST):
    def __init__(self, *args, **kwargs) -> None:
        self.input_as_label = kwargs.pop('input_as_label')
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int) -> tuple[Data, Data]:
        x, t = super().__getitem__(idx)
        if self.input_as_label:
            return Data(x), Data(x)
        return Data(x), Data(t)


def create_mnist_dataset(
    opt: MNISTDatasetOption,
    transform: Transform,
    is_train: bool,
) -> Dataset:
    return MNIST(
        opt.root,
        input_as_label=opt.input_as_label,
        train=is_train,
        download=True,
        transform=transform,
    )
