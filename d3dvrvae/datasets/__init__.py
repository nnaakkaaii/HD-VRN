from torch.utils.data import Dataset

from .option import DatasetOption
from .mnist import MNISTDatasetOption, create_mnist_dataset
from ..transforms import Transform


def create_dataset(opt: DatasetOption,
                   transform: Transform,
                   is_train: bool,
                   ) -> Dataset:
    if isinstance(opt, MNISTDatasetOption):
        return create_mnist_dataset(opt, transform, is_train)
    raise NotImplementedError(f'dataset {opt.__class__.__name__} not implemented')
