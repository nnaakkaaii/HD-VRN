from dataclasses import dataclass

from torchvision import transforms

from .option import TransformOption
from .typing import Transform
from .normalization import MinMaxNormalizationOption, MinMaxNormalization
from .pool import Pool3dOption, Pool3d
from .random_shift import RandomShift3dOption, RandomShift3d
from .uniform_shape import UniformShape3dOption, UniformShape3d


@dataclass
class BasicTransform2dOption(TransformOption):
    normalize: bool = True
    pad_size: int | None = None
    crop_size: int | None = None


def create_basic_transform2d(opt: BasicTransform2dOption) -> Transform:
    ts = [transforms.ToTensor()]
    if opt.pad_size:
        ts.append(transforms.Pad(opt.pad_size, fill=0))
    if opt.normalize:
        ts.append(transforms.Normalize((0.5,), (0.5,)))
    if opt.crop_size:
        ts.append(transforms.RandomCrop(opt.crop_size))
    return transforms.Compose(ts)


@dataclass
class BasicTransform3dOption(
    TransformOption,
    MinMaxNormalizationOption,
    Pool3dOption,
    RandomShift3dOption,
    UniformShape3dOption,
):
    use_random_shift: bool = True


def create_basic_transform3d(opt: BasicTransform3dOption, is_train: bool) -> Transform:
    ts = [MinMaxNormalization()]
    if is_train and opt.use_random_shift:
        ts.append(RandomShift3d(opt.max_shifts))
    ts.append(UniformShape3d(opt.target_shape))
    if opt.pool_size != [1, 1, 1]:
        ts.append(Pool3d(opt.pool_size))
    return transforms.Compose(ts)
