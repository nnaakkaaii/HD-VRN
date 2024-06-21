from torch import Tensor, cat


def optimize_output(
    x_2d: Tensor,
    x_2d_0: Tensor,
    x_2d_t: Tensor,
    x_2d_all: Tensor,
    x_3d: Tensor,
    x_3d_0: Tensor,
    x_3d_t: Tensor,
    x_3d_all: Tensor,
    content_phase: str,
    motion_phase: str,
    motion_aggregation: str,
    pred_diff: bool = False,
) -> dict[str, Tensor]:
    if content_phase == "all":
        xp_0 = x_3d_all
    elif content_phase == "0":
        xp_0 = x_3d_0
    elif content_phase == "t":
        xp_0 = x_3d_t
    else:
        raise KeyError(f"unknown content phase {content_phase}")

    xp = x_3d
    if pred_diff:
        xp = xp - xp_0.unsqueeze(0)

    xm_0 = x_2d_0  # pseudo
    if motion_phase == "0":
        pass
    elif motion_phase == "t":
        xm_0 = x_2d_t
    elif motion_phase == "all":
        xm_0 = x_2d_all
    else:
        raise KeyError(f"unknown motion phase {motion_phase}")

    xm = x_2d
    if motion_aggregation == "none":
        pass
    elif motion_aggregation == "diff":
        xm -= xm_0.unsqueeze(0)
    elif motion_aggregation == "concat":
        new = (xm.size(0),) + (1,) * xm_0.dim()
        xm = cat([xm, xm_0.unsqueeze(0).repeat(new)], dim=1)

    return {
        "xm": xm,  # (n, s, d, h)
        "xm_0": xm_0,  # (s, d, h)
        "xp": xp,  # (n, c, d, h, w)
        "xp_0": xp_0,  # (c | 2 * c, d, h, w)
    }
