from dataclasses import dataclass

from torchvision import transforms

from .option import TransformOption
from .typing import Transform


@dataclass
class BasicTransformOption(TransformOption):
    normalize: bool = True
    pad_size: int | None = None
    crop_size: int | None = None


def create_basic_transform(opt: BasicTransformOption) -> Transform:
    ts = []
    if opt.pad_size:
        ts.append(transforms.Pad(opt.pad_size, fill=0))
    ts.append([transforms.ToTensor()])
    if opt.normalize:
        ts.append(transforms.Normalize((0.5,), (0.5,)))
    if opt.crop_size:
        ts.append(transforms.RandomCrop(opt.crop_size))
    return transforms.Compose(ts)
