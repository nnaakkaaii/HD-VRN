from dataclasses import dataclass

from torchvision import transforms
from omegaconf import MISSING

from .option import TransformOption
from .typing import Transform
from .normalization import MinMaxNormalizationOption, MinMaxNormalization
from .random_shift import RandomShift3dOption, create_random_shift3d
from .uniform_shape import UniformShape3dOption, create_uniform_shape3d
from .pool import Pool3dOption, create_pool3d


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
    if isinstance(opt, ToTensorOption):
        return transforms.ToTensor()
    if isinstance(opt, Pad2dOption):
        return transforms.Pad(opt.pad_size)
    if isinstance(opt, Normalize2dOption):
        return transforms.Normalize((opt.mean,), (opt.std,))
    if isinstance(opt, Crop2dOption):
        return transforms.RandomCrop(opt.crop_size)
    if isinstance(opt, MinMaxNormalizationOption):
        return MinMaxNormalization()
    if isinstance(opt, RandomShift3dOption):
        return create_random_shift3d(opt)
    if isinstance(opt, UniformShape3dOption):
        return create_uniform_shape3d(opt)
    if isinstance(opt, Pool3dOption):
        return create_pool3d(opt)
    raise NotImplementedError(f"{opt.__class__.__name__} is not implemented")
