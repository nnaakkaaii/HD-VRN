from torch import randn

from hrdae.models.networks.r_ae import (
    Decoder2d,
    Decoder3d,
    RAE2d,
    RAE3d,
)
from hrdae.models.networks.motion_encoder import (
    MotionNormalEncoder1d,
    MotionNormalEncoder2d,
)


def test_decoder2d():
    b, n, c_, h, w = 8, 10, 1, 16, 16
    latent = 32

    m = randn((b * n, latent, h // 4, w // 4))
    net = Decoder2d(
        c_,
        latent,
        [
            {
                "kernel_size": [3],
                "stride": [2],
                "padding": [1],
                "output_padding": [1],
            }
        ]
        * 2,
        debug_show_dim=False,
    )
    x = net(m)
    assert x.size() == (b * n, c_, h, w)


def test_decoder3d():
    b, n, c_, d, h, w = 8, 10, 1, 16, 16, 16
    latent = 32

    m = randn((b * n, latent, d // 4, h // 4, w // 4))
    net = Decoder3d(
        c_,
        latent,
        [
            {
                "kernel_size": [3],
                "stride": [2],
                "padding": [1],
                "output_padding": [1],
            }
        ]
        * 2,
        debug_show_dim=False,
    )
    x = net(m)
    assert x.size() == (b * n, c_, d, h, w)


def test_rae2d():
    b, n, c, s, h, w = 8, 10, 1, 3, 16, 16
    latent = 32

    net = RAE2d(
        c,
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
        upsample_size=[h // 4, w // 4],
    )
    out = net(
        randn((b, n, s, h)),
        randn((b, 2, h, w)),
    )
    assert out.size() == (b, n, c, h, w)


def test_rae3d():
    b, n, c, s, d, h, w = 8, 10, 1, 3, 16, 16, 16
    latent = 32

    net = RAE3d(
        c,
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
        upsample_size=[d // 4, h // 4, w // 4],
    )
    out = net(
        randn((b, n, s, d, h)),
        randn((b, 2, d, h, w)),
    )
    assert out.size() == (b, n, c, d, h, w)
