from dataclasses import dataclass

from torchvision import transforms

from .typing import Transform
from .option import TransformOption


@dataclass
class BasicTransformOption(TransformOption):
    normalize: bool = True


def create_basic_transform(opt: BasicTransformOption) -> Transform:
    ts = [transforms.ToTensor()]
    if opt.normalize:
        ts.append(transforms.Normalize((0.5,), (0.5,)))
    return transforms.Compose(ts)
