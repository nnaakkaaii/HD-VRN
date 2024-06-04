from dataclasses import dataclass

from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets

from ..transforms import Transform
from .option import DatasetOption


@dataclass
class MNISTDatasetOption(DatasetOption):
    root: str = "data"
    input_as_label: bool = True


class MNIST(datasets.MNIST):
    def __init__(self, *args, **kwargs) -> None:
        self.input_as_label = kwargs.pop('input_as_label')
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        x, t = super().__getitem__(idx)
        if self.input_as_label:
            return {
                'x': x,
                't': x,
            }
        return {
            'x': x,
            't': t,
        }


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
