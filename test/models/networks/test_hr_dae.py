from torch import randn

from hrdae.models.networks.hr_dae import (
    HierarchicalContentEncoder2d,
    HierarchicalDecoder2d,
    HierarchicalContentEncoder3d,
    HierarchicalDecoder3d,
    HRDAE2d,
    HRDAE3d,
)
from hrdae.models.networks.motion_encoder import (
    MotionNormalEncoder1d,
    MotionNormalEncoder2d,
    MotionRNNEncoder2d,
)
from hrdae.models.networks.rnn import (
    TCN2d,
)


def test_hierarchical_content_encoder2d():
    b, c, h, w = 8, 1, 16, 16
    latent = 32

    x = randn((b, c, h, w))
    net = HierarchicalContentEncoder2d(
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
    c, cs = net(x)
    assert c.size() == (b, latent, h // 4, w // 4)
    assert len(cs) == 1
    assert cs[0].size() == (b, latent, h // 2, w // 2)


def test_hierarchical_content_encoder3d():
    b, c, d, h, w = 8, 1, 16, 16, 16
    latent = 32

    x = randn((b, c, d, h, w))
    net = HierarchicalContentEncoder3d(
        c,
        latent,
        [
            {
                "kernel_size": [3],
                "stride": [2],
                "padding": [1],
            },
            {
                "kernel_size": [3],
                "stride": [2],
                "padding": [1],
            },
            {
                "kernel_size": [3],
                "stride": [1],
                "padding": [1],
            },
        ],
        debug_show_dim=False,
    )
    c, cs = net(x)
    assert c.size() == (b, latent, d // 4, h // 4, w // 4)
    assert len(cs) == 2
    assert cs[0].size() == (b, latent, d // 2, h // 2, w // 2)
    assert cs[1].size() == (b, latent, d // 4, h // 4, w // 4)


def test_hierarchical_decoder2d__concat():
    b, n, c_, h, w = 8, 10, 1, 16, 16
    latent = 32

    m = randn((b * n, latent, h))
    c = randn((b * n, latent, h // 4, w // 4))
    cs = [
        randn((b * n, latent, h // 4, w // 4)),
        randn((b * n, latent, h // 2, w // 2)),
    ]
    net = HierarchicalDecoder2d(
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
        aggregation_method="concat",
        debug_show_dim=False,
    )
    x = net(m, c, cs)
    assert x.size() == (b * n, c_, h, w)


def test_hierarchical_decoder2d__sum():
    b, n, c_, h, w = 8, 10, 1, 16, 16
    latent = 32

    m = randn((b * n, latent, h))
    c = randn((b * n, latent, h // 4, w // 4))
    cs = [
        randn((b * n, latent, h // 4, w // 4)),
        randn((b * n, latent, h // 2, w // 2)),
    ]
    net = HierarchicalDecoder2d(
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
        aggregation_method="sum",
        debug_show_dim=False,
    )
    x = net(m, c, cs)
    assert x.size() == (b * n, 1, h, w)


def test_hierarchical_decoder3d__concat():
    b, n, c_, d, h, w = 8, 10, 1, 16, 16, 16
    latent = 32

    m = randn((b * n, latent, d, h))
    c = randn((b * n, latent, h // 4, h // 4, w // 4))
    cs = [
        randn((b * n, latent, h // 4, h // 4, w // 4)),
        randn((b * n, latent, h // 2, h // 2, w // 2)),
    ]
    net = HierarchicalDecoder3d(
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
        aggregation_method="concat",
        debug_show_dim=False,
    )
    x = net(m, c, cs)
    assert x.size() == (b * n, c_, d, h, w)


def test_hierarchical_decoder3d__sum():
    b, n, c_, d, h, w = 8, 10, 1, 16, 16, 16
    latent = 32

    m = randn((b * n, latent, d, h))
    c = randn((b * n, latent, d // 4, h // 4, w // 4))
    cs = [
        randn((b * n, latent, d // 4, h // 4, w // 4)),
        randn((b * n, latent, d // 2, h // 2, w // 2)),
    ]
    net = HierarchicalDecoder3d(
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
        aggregation_method="sum",
        debug_show_dim=False,
    )
    x = net(m, c, cs)
    assert x.size() == (b * n, c_, d, h, w)


def test_hrdae2d():
    b, n, c, s, h, w = 8, 10, 1, 3, 16, 16
    latent = 32

    net = HRDAE2d(
        2,
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
        activation="sigmoid",
    )
    out = net(
        randn((b, n, s, h)),
        randn((b, 2, h, w)),
    )
    assert out.size() == (b, n, c, h, w)


def test_hrdae3d():
    b, n, c, s, d, h, w = 8, 10, 1, 3, 16, 16, 16
    latent = 32

    net = HRDAE3d(
        2,
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
        activation="sigmoid",
        )
    out = net(
        randn((b, n, s, d, h)),
        randn((b, 2, d, h, w)),
    )
    assert out.size() == (b, n, c, d, h, w)


def test_hrdae3d__tcn2d():
    b, n, c, s, d, h, w = 8, 10, 1, 3, 8, 16, 16
    latent = 32

    net = HRDAE3d(
        2,
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
        motion_encoder=MotionRNNEncoder2d(
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
            TCN2d(
                latent,
                2,
                (2, 4),
                3,
            )
        ),
        activation="sigmoid",
    )
    out = net(
        randn((b, n, s, d, h)),
        randn((b, 2, d, h, w)),
    )
    assert out.size() == (b, n, c, d, h, w)
