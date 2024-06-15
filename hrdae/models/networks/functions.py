from torch import Tensor, cat
from torch.nn.functional import interpolate


def aggregate(m: Tensor, c: Tensor, method: str) -> Tensor:
    if method == "sum":
        return c + _upsample_motion_tensor(m, c)
    elif method == "concat":
        return cat([c, _upsample_motion_tensor(m, c)], dim=1)
    raise NotImplementedError(f"{method} not implemented")


def _upsample_motion_tensor(m: Tensor, c: Tensor) -> Tensor:
    if c.dim() == 5:
        return _upsample_motion_tensor3d(m, c)
    elif c.dim() == 4:
        return _upsample_motion_tensor2d(m, c)
    raise ValueError(f"Invalid dimension {c.dim()}")


def _upsample_motion_tensor2d(m: Tensor, c: Tensor) -> Tensor:
    b, c_, h, w = c.size()
    m = m.unsqueeze(-1)
    m = interpolate(m, size=(h, w), mode="bilinear", align_corners=True)
    return m


def _upsample_motion_tensor3d(m: Tensor, c: Tensor) -> Tensor:
    b, c_, d, h, w = c.size()
    m = m.unsqueeze(-1)
    m = interpolate(m, size=(d, h, w), mode="trilinear", align_corners=True)
    return m
