from torch.utils.data import Dataset

from ..transforms import Transform
from .mnist import MNISTDatasetOption, create_mnist_dataset
from .option import DatasetOption


def create_dataset(
    opt: DatasetOption,
    transform: Transform,
    is_train: bool,
) -> Dataset:
    if isinstance(opt, MNISTDatasetOption):
        return create_mnist_dataset(opt, transform, is_train)
    raise NotImplementedError(f"dataset {opt.__class__.__name__} not implemented")
