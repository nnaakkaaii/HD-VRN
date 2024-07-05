from dataclasses import dataclass

from omegaconf import MISSING
from torchvision import transforms

from .normalization import MinMaxNormalization, MinMaxNormalizationOption
from .option import TransformOption
from .pool import (
    Pool2dOption,
    Pool3dOption,
    create_pool2d,
    create_pool3d,
)
from .random_shift import (
    RandomShift2dOption,
    RandomShift3dOption,
    create_random_shift2d,
    create_random_shift3d,
)
from .typing import Transform
from .uniform_shape import UniformShape3dOption, create_uniform_shape3d


@dataclass
class ToTensorOption(TransformOption):
    pass


@dataclass
class Pad2dOption(TransformOption):
    pad_size: int = MISSING


@dataclass
class Normalize2dOption(TransformOption):
    mean: float = 0.5
    std: float = 0.5


@dataclass
class Crop2dOption(TransformOption):
    crop_size: int = MISSING


def create_transform(opt: TransformOption) -> Transform:
    if isinstance(opt, ToTensorOption) and type(opt) is ToTensorOption:
        return transforms.ToTensor()
    if isinstance(opt, Pad2dOption) and type(opt) is Pad2dOption:
        return transforms.Pad(opt.pad_size)
    if isinstance(opt, Normalize2dOption) and type(opt) is Normalize2dOption:
        return transforms.Normalize((opt.mean,), (opt.std,))
    if isinstance(opt, Crop2dOption) and type(opt) is Crop2dOption:
        return transforms.RandomCrop(opt.crop_size)
    if (
        isinstance(opt, MinMaxNormalizationOption)
        and type(opt) is MinMaxNormalizationOption
    ):
        return MinMaxNormalization()
    if isinstance(opt, RandomShift2dOption) and type(opt) is RandomShift2dOption:
        return create_random_shift2d(opt)
    if isinstance(opt, RandomShift3dOption) and type(opt) is RandomShift3dOption:
        return create_random_shift3d(opt)
    if isinstance(opt, UniformShape3dOption) and type(opt) is UniformShape3dOption:
        return create_uniform_shape3d(opt)
    if isinstance(opt, Pool2dOption) and type(opt) is Pool2dOption:
        return create_pool2d(opt)
    if isinstance(opt, Pool3dOption) and type(opt) is Pool3dOption:
        return create_pool3d(opt)
    raise NotImplementedError(f"{opt.__class__.__name__} is not implemented")
