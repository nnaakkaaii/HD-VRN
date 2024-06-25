from torch import Tensor
from torch.nn.functional import interpolate


def upsample_motion_tensor(m: Tensor, c: Tensor) -> Tensor:
    if c.dim() == 5:
        return _upsample_motion_tensor3d(m, c)
    elif c.dim() == 4:
        return _upsample_motion_tensor2d(m, c)
    raise ValueError(f"Invalid dimension {c.dim()}")


def _upsample_motion_tensor2d(m: Tensor, c: Tensor) -> Tensor:
    b, c_, h, w = c.size()
    m = interpolate(m, size=(h, w), mode="bilinear", align_corners=True)
    return m


def _upsample_motion_tensor3d(m: Tensor, c: Tensor) -> Tensor:
    b, c_, d, h, w = c.size()
    m = interpolate(m, size=(d, h, w), mode="trilinear", align_corners=True)
    return m
