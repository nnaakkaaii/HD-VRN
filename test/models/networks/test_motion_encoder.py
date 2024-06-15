from torch import randn

from hrdae.models.networks.motion_encoder import (
    MotionNormalEncoder1d,
    MotionNormalEncoder2d,
    MotionRNNEncoder1d,
    MotionRNNEncoder2d,
    MotionConv2dEncoder1d,
    MotionConv3dEncoder2d,
    MotionGuidedEncoder1d,
    MotionGuidedEncoder2d,
    MotionTSNEncoder1d,
    MotionTSNEncoder2d,
)
from hrdae.models.networks.rnn import GRU1d, GRU2d


def test_motion_normal_encoder1d():
    b, n, c, h = 8, 10, 1, 16
    latent = 32

    x = randn((b, n, c, h))
    net = MotionNormalEncoder1d(
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
        debug_show_dim=False,
    )
    c = net(x)
    assert c.size() == (b, n, latent, h // 4)


def test_motion_normal_encoder2d():
    b, n, c, d, h = 8, 10, 1, 16, 16
    latent = 32

    x = randn((b, n, c, d, h))
    net = MotionNormalEncoder2d(
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
        debug_show_dim=False,
    )
    c = net(x)
    assert c.size() == (b, n, latent, d // 4, h // 4)


def test_motion_rnn_encoder1d():
    b, n, c, h = 8, 10, 1, 16
    latent, layer = 32, 2

    x = randn((b, n, c, h))
    net = MotionRNNEncoder1d(
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
        GRU1d(latent, layer, h // 4),
        debug_show_dim=False,
    )
    c = net(x)
    assert c.size() == (b, n, latent, h // 4)


def test_motion_rnn_encoder2d():
    b, n, c, d, h = 8, 10, 1, 16, 16
    latent, layer = 32, 2

    x = randn((b, n, c, d, h))
    net = MotionRNNEncoder2d(
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
        GRU2d(latent, layer, [d // 4, h // 4]),
        debug_show_dim=False,
    )
    c = net(x)
    assert c.size() == (b, n, latent, d // 4, h // 4)


def test_motion_conv2d_encoder1d():
    b, n, c, h = 8, 10, 1, 16
    latent, layer = 32, 2

    x = randn((b, n, c, h))
    net = MotionConv2dEncoder1d(
        c,
        latent,
        [
            {
                "kernel_size": [3],
                "stride": [1, 2],
                "padding": [1],
            }
        ]
        * 2,
        debug_show_dim=False,
    )
    c = net(x)
    assert c.size() == (b, n, latent, h // 4)


def test_motion_conv3d_encoder2d():
    b, n, c, d, h = 8, 10, 1, 16, 16
    latent, layer = 32, 2

    x = randn((b, n, c, d, h))
    net = MotionConv3dEncoder2d(
        c,
        latent,
        [
            {
                "kernel_size": [3],
                "stride": [1, 2, 2],
                "padding": [1],
            }
        ]
        * 2,
        debug_show_dim=False,
    )
    c = net(x)
    assert c.size() == (b, n, latent, d // 4, h // 4)


def test_motion_guided_encoder1d():
    b, n, c, h = 8, 10, 1, 16
    latent = 32

    net = MotionGuidedEncoder1d(
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
        debug_show_dim=False,
    )
    c = net(
        randn((b, n, c, h)),
        randn((b, c, h)),
    )
    assert c.size() == (b, n, latent, h // 4)


def test_motion_guided_encoder2d():
    b, n, c, d, h = 8, 10, 1, 16, 16
    latent = 32

    net = MotionGuidedEncoder2d(
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
        debug_show_dim=False,
    )
    c = net(
        randn((b, n, c, d, h)),
        randn((b, c, d, h)),
    )
    assert c.size() == (b, n, latent, d // 4, h // 4)


def test_motion_tsn_encoder1d():
    b, n, c, h = 8, 10, 1, 16
    latent = 32

    net = MotionTSNEncoder1d(
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
        debug_show_dim=False,
    )
    c = net(
        randn((b, n, c, h)),
        randn((b, c, h)),
    )
    assert c.size() == (b, n, latent, h // 4)


def test_motion_tsn_encoder2d():
    b, n, c, d, h = 8, 10, 1, 16, 16
    latent = 32

    net = MotionTSNEncoder2d(
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
        debug_show_dim=False,
    )
    c = net(
        randn((b, n, c, d, h)),
        randn((b, c, d, h)),
    )
    assert c.size() == (b, n, latent, d // 4, h // 4)
