from torch import randn

from hrdae.models.networks.r_dae import (
    RDAE2d,
    RDAE3d,
    CycleRDAE2d,
    CycleRDAE3d,
)
from hrdae.models.networks.motion_encoder import (
    MotionNormalEncoder1d,
    MotionNormalEncoder2d,
)


def test_rdae2d():
    b, n, c, s, h, w = 8, 10, 1, 3, 16, 16
    hidden = 16
    latent = 4

    net = RDAE2d(
        2,
        c,
        hidden,
        latent,
        [
            {
                "kernel_size": [3],
                "stride": [2],
                "padding": [1],
            }
        ]
        * 2,
        motion_encoder=MotionNormalEncoder1d(
            s,
            hidden,
            latent,
            [
                {
                    "kernel_size": [3],
                    "stride": [2],
                    "padding": [1],
                }
            ]
            * 2,
        ),
        aggregator="addition",
        activation="sigmoid",
    )
    out, _ = net(
        randn((b, n, s, h)),
        randn((b, 2, h, w)),
    )
    assert out.size() == (b, n, c, h, w)


def test_cycle_rdae2d():
    b, n, c, s, h, w = 8, 10, 1, 3, 16, 16
    hidden = 16
    latent = 4

    net = CycleRDAE2d(
        1,
        c,
        hidden,
        latent,
        [
            {
                "kernel_size": [3],
                "stride": [2],
                "padding": [1],
            }
        ]
        * 2,
        motion_encoder=MotionNormalEncoder1d(
            s,
            hidden,
            latent,
            [
                {
                    "kernel_size": [3],
                    "stride": [2],
                    "padding": [1],
                }
            ]
            * 2,
        ),
        aggregator="addition",
        activation="sigmoid",
    )
    out, cs = net(
        randn((b, n, s, h)),
        randn((b, 1, h, w)),
    )
    assert out.size() == (b, n, c, h, w)
    assert cs[0].size() == (b, n, latent, h // 4, w // 4)


def test_rdae3d():
    b, n, c, s, d, h, w = 8, 10, 1, 3, 16, 16, 16
    hidden = 16
    latent = 4

    net = RDAE3d(
        2,
        c,
        hidden,
        latent,
        [
            {
                "kernel_size": [3],
                "stride": [2],
                "padding": [1],
            }
        ]
        * 2,
        motion_encoder=MotionNormalEncoder2d(
            s,
            hidden,
            latent,
            [
                {
                    "kernel_size": [3],
                    "stride": [2],
                    "padding": [1],
                }
            ]
            * 2,
        ),
        aggregator="addition",
        activation="sigmoid",
    )
    out, _ = net(
        randn((b, n, s, d, h)),
        randn((b, 2, d, h, w)),
    )
    assert out.size() == (b, n, c, d, h, w)


def test_cycle_rdae3d():
    b, n, c, s, d, h, w = 8, 10, 1, 3, 16, 16, 16
    hidden = 16
    latent = 4

    net = CycleRDAE3d(
        1,
        c,
        hidden,
        latent,
        [
            {
                "kernel_size": [3],
                "stride": [2],
                "padding": [1],
            }
        ]
        * 2,
        motion_encoder=MotionNormalEncoder2d(
            s,
            hidden,
            latent,
            [
                {
                    "kernel_size": [3],
                    "stride": [2],
                    "padding": [1],
                }
            ]
            * 2,
        ),
        aggregator="addition",
        activation="sigmoid",
    )
    out, cs = net(
        randn((b, n, s, d, h)),
        randn((b, 1, d, h, w)),
    )
    assert out.size() == (b, n, c, d, h, w)
    assert cs[0].size() == (b, n, latent, d // 4, h // 4, w // 4)
