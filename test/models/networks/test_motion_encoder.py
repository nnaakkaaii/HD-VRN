from torch import randn

from hrdae.models.networks.motion_encoder import (
    MotionConv2dEncoder1d,
    MotionConv3dEncoder2d,
    MotionNormalEncoder1d,
    MotionNormalEncoder2d,
    MotionRNNEncoder1d,
    MotionRNNEncoder2d,
)
from hrdae.models.networks.rnn import GRU1d, GRU2d


def test_motion_normal_encoder1d():
    b, n, c, h, w = 8, 10, 1, 16, 16
    hidden = 16
    latent = 4

    x = randn((b, n, c, h))
    net = MotionNormalEncoder1d(
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
        [
            {
                "kernel_size": [3],
                "stride": [1, 2],
                "padding": [1],
                "output_padding": [0, 1],
            }
        ]
        * 2,
        debug_show_dim=False,
    )
    c = net(x)
    assert c.size() == (b, n, latent, h // 4, w // 4)


def test_motion_normal_encoder2d():
    b, n, c, d, h, w = 8, 10, 1, 16, 16, 16
    hidden = 16
    latent = 4

    x = randn((b, n, c, d, h))
    net = MotionNormalEncoder2d(
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
        [
            {
                "kernel_size": [3],
                "stride": [1, 1, 2],
                "padding": [1],
                "output_padding": [0, 0, 1],
            }
        ]
        * 2,
        debug_show_dim=False,
    )
    c = net(x)
    assert c.size() == (b, n, latent, d // 4, h // 4, w // 4)


def test_motion_rnn_encoder1d():
    b, n, c, h, w = 8, 10, 1, 16, 16
    layer = 2
    hidden = 16
    latent = 4

    x = randn((b, n, c, h))
    net = MotionRNNEncoder1d(
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
        [
            {
                "kernel_size": [3],
                "stride": [1, 2],
                "padding": [1],
                "output_padding": [0, 1],
            }
        ]
        * 2,
        GRU1d(hidden, layer, h // 4),
        debug_show_dim=False,
    )
    c = net(x)
    assert c.size() == (b, n, latent, h // 4, w // 4)


def test_motion_rnn_encoder2d():
    b, n, c, d, h, w = 8, 10, 1, 16, 16, 16
    layer = 2
    hidden = 16
    latent = 4

    x = randn((b, n, c, d, h))
    net = MotionRNNEncoder2d(
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
        [
            {
                "kernel_size": [3],
                "stride": [1, 1, 2],
                "padding": [1],
                "output_padding": [0, 0, 1],
            }
        ]
        * 2,
        GRU2d(hidden, layer, [d // 4, h // 4]),
        debug_show_dim=False,
    )
    c = net(x)
    assert c.size() == (b, n, latent, d // 4, h // 4, w // 4)


def test_motion_conv2d_encoder1d():
    b, n, c, h, w = 8, 10, 1, 16, 16
    hidden = 16
    latent = 4

    x = randn((b, n, c, h))
    net = MotionConv2dEncoder1d(
        c,
        hidden,
        latent,
        [
            {
                "kernel_size": [3],
                "stride": [1, 2],
                "padding": [1],
            }
        ]
        * 2,
        [
            {
                "kernel_size": [3],
                "stride": [1, 2],
                "padding": [1],
                "output_padding": [0, 1],
            }
        ]
        * 2,
        debug_show_dim=False,
    )
    c = net(x)
    assert c.size() == (b, n, latent, h // 4, w // 4)


def test_motion_conv3d_encoder2d():
    b, n, c, d, h, w = 8, 10, 1, 16, 16, 16
    hidden = 16
    latent = 4

    x = randn((b, n, c, d, h))
    net = MotionConv3dEncoder2d(
        c,
        hidden,
        latent,
        [
            {
                "kernel_size": [3],
                "stride": [1, 2, 2],
                "padding": [1],
            }
        ]
        * 2,
        [
            {
                "kernel_size": [3],
                "stride": [1, 1, 2],
                "padding": [1],
                "output_padding": [0, 0, 1],
            }
        ]
        * 2,
        debug_show_dim=False,
    )
    c = net(x)
    assert c.size() == (b, n, latent, d // 4, h // 4, w // 4)
