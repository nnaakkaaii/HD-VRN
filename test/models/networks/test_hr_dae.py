from torch import randn

from hrdae.models.networks import create_network, HRDAE2dOption, HRDAE3dOption
from hrdae.models.networks.hr_dae import (
    HierarchicalEncoder2d,
    HierarchicalDecoder2d,
    HierarchicalEncoder3d,
    HierarchicalDecoder3d,
)
from hrdae.models.networks.motion_encoder import (
    MotionNormalEncoder1dOption,
    MotionNormalEncoder2dOption,
)


def test_hierarchical_content_encoder2d():
    b, c, h, w = 8, 1, 16, 16
    hidden = 16
    latent = 4

    x = randn((b, c, h, w))
    net = HierarchicalEncoder2d(
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
        debug_show_dim=False,
    )
    c, cs = net(x)
    assert c.size() == (b, latent, h // 4, w // 4)
    assert len(cs) == 1
    assert cs[0].size() == (b, hidden, h // 2, w // 2)


def test_hierarchical_content_encoder3d():
    b, c, d, h, w = 8, 1, 16, 16, 16
    hidden = 16
    latent = 4

    x = randn((b, c, d, h, w))
    net = HierarchicalEncoder3d(
        c,
        hidden,
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
    assert cs[0].size() == (b, hidden, d // 2, h // 2, w // 2)
    assert cs[1].size() == (b, hidden, d // 4, h // 4, w // 4)


def test_hierarchical_decoder2d():
    b, n, c_, h, w = 8, 10, 1, 16, 16
    hidden = 16
    latent = 4

    x = randn((b * n, latent, h // 4, w // 4))
    cs = [
        randn((b * n, hidden, h // 4, w // 4)),
        randn((b * n, hidden, h // 2, w // 2)),
    ]
    net = HierarchicalDecoder2d(
        c_,
        hidden,
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
        aggregation="concatenation",
        debug_show_dim=False,
    )
    x = net(x, cs)
    assert x.size() == (b * n, c_, h, w)


def test_hierarchical_decoder3d():
    b, n, c_, d, h, w = 8, 10, 1, 16, 16, 16
    hidden = 16
    latent = 4

    x = randn((b * n, latent, d // 4, h // 4, w // 4))
    cs = [
        randn((b * n, hidden, d // 4, h // 4, w // 4)),
        randn((b * n, hidden, d // 2, h // 2, w // 2)),
    ]
    net = HierarchicalDecoder3d(
        c_,
        hidden,
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
        aggregation="concatenation",
        debug_show_dim=False,
    )
    x = net(x, cs)
    assert x.size() == (b * n, c_, d, h, w)


def test_hrdae2d():
    b, n, s, h, w = 8, 10, 3, 16, 16
    hidden = 16
    latent = 4

    opt = HRDAE2dOption(
        in_channels=2,
        hidden_channels=hidden,
        latent_dim=latent,
        conv_params=[
            {
                "kernel_size": [3],
                "stride": [2],
                "padding": [1],
            }
        ]
        * 2,
        motion_encoder=MotionNormalEncoder1dOption(
            in_channels=s,
            hidden_channels=hidden,
            conv_params=[
                {
                    "kernel_size": [3],
                    "stride": [2],
                    "padding": [1],
                }
            ]
            * 2,
            deconv_params=[
                {
                    "kernel_size": [3],
                    "stride": [1, 2],
                    "padding": [1],
                    "output_padding": [0, 1],
                }
            ]
            * 2,
        ),
        aggregator="addition",
        activation="sigmoid",
    )
    net = create_network(1, opt)
    out, cs, ds = net(
        randn((b, n, s, h)),
        randn((b, 2, h, w)),
    )
    assert out.size() == (b, n, 1, h, w)
    assert len(cs) == 3
    assert cs[0].size() == (b, latent, h // 4, w // 4)
    assert cs[1].size() == (b, hidden, h // 2, w // 2)
    assert cs[2].size() == (b, hidden, h // 4, w // 4)
    assert len(ds) == 0


def test_hrdae2d__concatenation():
    b, n, s, h, w = 8, 10, 3, 16, 16
    hidden = 16
    latent = 4

    opt = HRDAE2dOption(
        in_channels=2,
        hidden_channels=hidden,
        latent_dim=latent,
        conv_params=[
            {
                "kernel_size": [3],
                "stride": [2],
                "padding": [1],
            }
        ]
        * 2,
        motion_encoder=MotionNormalEncoder1dOption(
            in_channels=s,
            hidden_channels=hidden,
            conv_params=[
                {
                    "kernel_size": [3],
                    "stride": [2],
                    "padding": [1],
                }
            ]
            * 2,
            deconv_params=[
                {
                    "kernel_size": [3],
                    "stride": [1, 2],
                    "padding": [1],
                    "output_padding": [0, 1],
                }
            ]
            * 2,
        ),
        aggregator="concatenation",
        activation="sigmoid",
    )
    net = create_network(1, opt)
    out, cs, ds = net(
        randn((b, n, s, h)),
        randn((b, 2, h, w)),
    )
    assert out.size() == (b, n, 1, h, w)
    assert len(cs) == 3
    assert cs[0].size() == (b, latent, h // 4, w // 4)
    assert cs[1].size() == (b, hidden, h // 2, w // 2)
    assert cs[2].size() == (b, hidden, h // 4, w // 4)
    assert len(ds) == 0


def test_hrdae3d():
    b, n, s, d, h, w = 8, 10, 3, 16, 16, 16
    hidden = 16
    latent = 4

    opt = HRDAE3dOption(
        in_channels=2,
        hidden_channels=hidden,
        latent_dim=latent,
        conv_params=[
            {
                "kernel_size": [3],
                "stride": [2],
                "padding": [1],
            }
        ]
        * 2,
        motion_encoder=MotionNormalEncoder2dOption(
            in_channels=s,
            hidden_channels=hidden,
            conv_params=[
                {
                    "kernel_size": [3],
                    "stride": [2],
                    "padding": [1],
                }
            ]
            * 2,
            deconv_params=[
                {
                    "kernel_size": [3],
                    "stride": [1, 1, 2],
                    "padding": [1],
                    "output_padding": [0, 0, 1],
                }
            ]
            * 2,
        ),
        aggregator="addition",
        activation="sigmoid",
    )
    net = create_network(1, opt)
    out, cs, ds = net(
        randn((b, n, s, d, h)),
        randn((b, 2, d, h, w)),
    )
    assert out.size() == (b, n, 1, d, h, w)
    assert len(cs) == 3
    assert cs[0].size() == (b, latent, d // 4, h // 4, w // 4)
    assert cs[1].size() == (b, hidden, d // 2, h // 2, w // 2)
    assert cs[2].size() == (b, hidden, d // 4, h // 4, w // 4)
    assert len(ds) == 0


def test_hrdae3d__concatenation():
    b, n, s, d, h, w = 8, 10, 3, 16, 16, 16
    hidden = 16
    latent = 4

    opt = HRDAE3dOption(
        in_channels=2,
        hidden_channels=hidden,
        latent_dim=latent,
        conv_params=[
            {
                "kernel_size": [3],
                "stride": [2],
                "padding": [1],
            }
        ]
        * 2,
        motion_encoder=MotionNormalEncoder2dOption(
            in_channels=s,
            hidden_channels=hidden,
            conv_params=[
                {
                    "kernel_size": [3],
                    "stride": [2],
                    "padding": [1],
                }
            ]
            * 2,
            deconv_params=[
                {
                    "kernel_size": [3],
                    "stride": [1, 1, 2],
                    "padding": [1],
                    "output_padding": [0, 0, 1],
                }
            ]
            * 2,
        ),
        aggregator="concatenation",
        activation="sigmoid",
    )
    net = create_network(1, opt)
    out, cs, ds = net(
        randn((b, n, s, d, h)),
        randn((b, 2, d, h, w)),
    )
    assert out.size() == (b, n, 1, d, h, w)
    assert len(cs) == 3
    assert cs[0].size() == (b, latent, d // 4, h // 4, w // 4)
    assert cs[1].size() == (b, hidden, d // 2, h // 2, w // 2)
    assert cs[2].size() == (b, hidden, d // 4, h // 4, w // 4)
    assert len(ds) == 0


def test_cycle_hrdae2d():
    b, n, s, h, w = 8, 10, 3, 16, 16
    hidden = 16
    latent = 4

    opt = HRDAE2dOption(
        in_channels=1,
        hidden_channels=hidden,
        latent_dim=latent,
        conv_params=[
            {
                "kernel_size": [3],
                "stride": [2],
                "padding": [1],
            }
        ]
        * 2,
        motion_encoder=MotionNormalEncoder1dOption(
            in_channels=s,
            hidden_channels=hidden,
            conv_params=[
                {
                    "kernel_size": [3],
                    "stride": [2],
                    "padding": [1],
                }
            ]
            * 2,
            deconv_params=[
                {
                    "kernel_size": [3],
                    "stride": [1, 2],
                    "padding": [1],
                    "output_padding": [0, 1],
                }
            ]
            * 2,
        ),
        aggregator="addition",
        activation="sigmoid",
        cycle=True,
    )
    net = create_network(1, opt)
    out, cs, ds = net(
        randn((b, n, s, h)),
        randn((b, 1, h, w)),
    )
    assert out.size() == (b, n, 1, h, w)
    assert len(cs) == 3
    assert cs[0].size() == (b, 1, latent, h // 4, w // 4)
    assert cs[1].size() == (b, 1, hidden, h // 2, w // 2)
    assert cs[2].size() == (b, 1, hidden, h // 4, w // 4)
    assert len(ds) == 3
    assert ds[0].size() == (b, n, latent, h // 4, w // 4)
    assert ds[1].size() == (b, n, hidden, h // 2, w // 2)
    assert ds[2].size() == (b, n, hidden, h // 4, w // 4)


def test_cycle_hrdae3d():
    b, n, s, d, h, w = 8, 10, 3, 16, 16, 16
    hidden = 16
    latent = 4

    opt = HRDAE3dOption(
        in_channels=1,
        hidden_channels=hidden,
        latent_dim=latent,
        conv_params=[
            {
                "kernel_size": [3],
                "stride": [2],
                "padding": [1],
            }
        ]
        * 2,
        motion_encoder=MotionNormalEncoder2dOption(
            in_channels=s,
            hidden_channels=hidden,
            conv_params=[
                {
                    "kernel_size": [3],
                    "stride": [2],
                    "padding": [1],
                }
            ]
            * 2,
            deconv_params=[
                {
                    "kernel_size": [3],
                    "stride": [1, 1, 2],
                    "padding": [1],
                    "output_padding": [0, 0, 1],
                }
            ]
            * 2,
        ),
        aggregator="addition",
        activation="sigmoid",
        cycle=True,
    )
    net = create_network(1, opt)
    out, cs, ds = net(
        randn((b, n, s, d, h)),
        randn((b, 1, d, h, w)),
    )
    assert out.size() == (b, n, 1, d, h, w)
    assert len(cs) == 3
    assert cs[0].size() == (b, 1, latent, d // 4, h // 4, w // 4)
    assert cs[1].size() == (b, 1, hidden, d // 2, h // 2, w // 2)
    assert cs[2].size() == (b, 1, hidden, d // 4, h // 4, w // 4)
    assert len(ds) == 3
    assert ds[0].size() == (b, n, latent, d // 4, h // 4, w // 4)
    assert ds[1].size() == (b, n, hidden, d // 2, h // 2, w // 2)
    assert ds[2].size() == (b, n, hidden, d // 4, h // 4, w // 4)
