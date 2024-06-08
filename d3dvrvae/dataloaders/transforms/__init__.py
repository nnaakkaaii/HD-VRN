from .basic import (
    BasicTransform2dOption, create_basic_transform2d,
    BasicTransform3dOption, create_basic_transform3d,
)
from .option import TransformOption
from .typing import Transform


def create_transform(opt: TransformOption, is_train: bool) -> Transform:
    if isinstance(opt, BasicTransform2dOption):
        return create_basic_transform2d(opt)
    if isinstance(opt, BasicTransform3dOption):
        return create_basic_transform3d(opt, is_train)
    raise NotImplementedError(f"{opt.__class__.__name__} is not implemented")
