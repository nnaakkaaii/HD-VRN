from torch.utils.data import Dataset

from ..transforms import Transform
from .ct import CT, CTDatasetOption, create_ct_dataset
from .mnist import MNISTDatasetOption, create_mnist_dataset
from .moving_mnist import MovingMNIST, MovingMNISTDatasetOption, create_moving_mnist_dataset
from .option import DatasetOption
from .seq_divide_wrapper import SeqDivideWrapper


def create_dataset(
    opt: DatasetOption,
    transform: Transform,
    is_train: bool,
) -> Dataset:
    if isinstance(opt, MNISTDatasetOption) and type(opt) is MNISTDatasetOption:
        return create_mnist_dataset(opt, transform, is_train)
    if (
        isinstance(opt, MovingMNISTDatasetOption)
        and type(opt) is MovingMNISTDatasetOption
    ):
        dataset = create_moving_mnist_dataset(opt, transform, is_train)
        if opt.sequential:
            return dataset
        return SeqDivideWrapper(dataset, MovingMNIST.PERIOD)
    if isinstance(opt, CTDatasetOption) and type(opt) is CTDatasetOption:
        dataset = create_ct_dataset(opt, transform, is_train)
        if opt.sequential:
            return dataset
        return SeqDivideWrapper(dataset, CT.PERIOD)
    raise NotImplementedError(f"dataset {opt.__class__.__name__} not implemented")
