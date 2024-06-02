from .basic import BasicTransformOption, create_basic_transform
from .option import TransformOption
from .typing import Transform


def create_transform(opt: TransformOption) -> Transform:
    if isinstance(opt, BasicTransformOption):
        return create_basic_transform(opt)
    raise NotImplementedError(f"{opt.__class__.__name__} is not implemented")
