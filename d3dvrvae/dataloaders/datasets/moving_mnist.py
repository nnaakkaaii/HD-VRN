from dataclasses import dataclass, field

from torch import Tensor
from torch.utils.data import Dataset
from torchvision import datasets

from ..transforms import Transform
from .option import DatasetOption


@dataclass
class MovingMNISTDatasetOption(DatasetOption):
    root: str = "data"
    slice_index: list[int] = field(default_factory=lambda: [16, 32, 48])


class MovingMNIST(datasets.MovingMNIST):
    def __init__(self, *args, **kwargs) -> None:
        self.slice_index = kwargs.pop('slice_index')
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        x, t = super().__getitem__(idx)
        return {
            "x": x,
            "t": t,
        }


def create_moving_mnist_dataset(
    opt: MovingMNISTDatasetOption,
    transform: Transform,
    is_train: bool,
) -> Dataset:
    return MovingMNIST(
        opt.root,
        slice_index=opt.slice_index,
        split="train" if is_train else "test",
        download=True,
        transform=transform,
    )


if __name__ == '__main__':
    def test():
        from torchvision import transforms
        from ..transforms import create_transform, ToTensorOption

        option = MovingMNISTDatasetOption(
            root="data",
        )
        dataset = create_moving_mnist_dataset(
            option,
            transform=transforms.Compose([
                create_transform(ToTensorOption()),
            ]),
            is_train=True,
        )
        data = dataset[0]
        for k, v in data.items():
            print(k, v.shape)

    test()
