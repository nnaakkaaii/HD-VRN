from dataclasses import dataclass, field

from torch import Tensor, min
from torch.nn.functional import pad

from .option import TransformOption
from .typing import Transform


@dataclass
class UniformShape3dOption(TransformOption):
    target_shape: list[int] = field(default_factory=lambda: [64, 512, 512])


def create_uniform_shape3d(opt: UniformShape3dOption) -> Transform:
    return UniformShape3d(opt.target_shape)


class UniformShape3d:
    def __init__(
        self,
        target_shape: list[int],
    ) -> None:
        self.target_shape = target_shape

    def __call__(
        self,
        x: Tensor,
    ) -> Tensor:
        _, n, h, w = x.size()
        target_n, target_h, target_w = self.target_shape

        min_val = float(min(x))

        # N dimension
        if n < target_n:
            pad_size = target_n - n
            pad_top = pad_size // 2
            pad_bottom = pad_size - pad_top
            x = pad(x, (0, 0, 0, 0, pad_top, pad_bottom), value=min_val)
        elif n > target_n:
            crop_size = n - target_n
            crop_top = crop_size // 2
            crop_bottom = crop_size - crop_top
            x = x[:, crop_top : n - crop_bottom, :, :]

        # H dimension
        if h < target_h:
            pad_size = target_h - h
            pad_top = pad_size // 2
            pad_bottom = pad_size - pad_top
            x = pad(x, (0, 0, pad_top, pad_bottom, 0, 0), value=min_val)
        elif h > target_h:
            crop_size = h - target_h
            crop_top = crop_size // 2
            crop_bottom = crop_size - crop_top
            x = x[:, :, crop_top : h - crop_bottom, :]

        # W dimension
        if w < target_w:
            pad_size = target_w - w
            pad_left = pad_size // 2
            pad_right = pad_size - pad_left
            x = pad(x, (pad_left, pad_right, 0, 0, 0, 0), value=min_val)
        elif w > target_w:
            crop_size = w - target_w
            crop_left = crop_size // 2
            crop_right = crop_size - crop_left
            x = x[:, :, :, crop_left : w - crop_right]

        return x
