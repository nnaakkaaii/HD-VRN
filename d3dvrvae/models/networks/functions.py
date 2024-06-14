from torch import Tensor, cat
from torch.nn.functional import interpolate


def aggregate(m: Tensor, c: Tensor, method: str) -> Tensor:
    if method == "sum":
        return c + _upsample_motion_tensor(m, c)
    elif method == "concat":
        return cat([c, _upsample_motion_tensor(m, c)], dim=1)
    raise NotImplementedError(f"{method} not implemented")


def _upsample_motion_tensor(m: Tensor, c: Tensor) -> Tensor:
    b, c_, d, h, w = c.size()
    m = m.unsqueeze(2)
    m = interpolate(m, size=(d, h, w), mode="trilinear", align_corners=True)
    return m
