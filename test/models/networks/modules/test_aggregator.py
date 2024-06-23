from torch import randn

from hrdae.models.networks.modules import create_aggregator2d, create_aggregator3d


def test_addition_aggregator2d():
    b, cc, cm, h, w = 8, 4, 8, 2, 2
    zc = randn((b, cc, h, w))
    zm = randn((b, cm, h, w))

    net = create_aggregator2d("addition", cc, cm)
    out = net((zc, zm))
    assert out.size() == (b, cc, h, w)


def test_addition_aggregator3d():
    b, cc, cm, d, h, w = 8, 4, 8, 2, 2, 2
    zc = randn((b, cc, d, h, w))
    zm = randn((b, cm, d, h, w))

    net = create_aggregator3d("addition", cc, cm)
    out = net((zc, zm))
    assert out.size() == (b, cc, d, h, w)


def test_multiplication_aggregator2d():
    b, cc, cm, h, w = 8, 4, 8, 2, 2
    zc = randn((b, cc, h, w))
    zm = randn((b, cm, h, w))

    net = create_aggregator2d("multiplication", cc, cm)
    out = net((zc, zm))
    assert out.size() == (b, cc, h, w)


def test_multiplication_aggregator3d():
    b, cc, cm, d, h, w = 8, 4, 8, 2, 2, 2
    zc = randn((b, cc, d, h, w))
    zm = randn((b, cm, d, h, w))

    net = create_aggregator3d("multiplication", cc, cm)
    out = net((zc, zm))
    assert out.size() == (b, cc, d, h, w)


def test_attention_aggregator2d():
    b, cc, cm, h, w = 8, 4, 8, 2, 2
    zc = randn((b, cc, h, w))
    zm = randn((b, cm, h, w))

    net = create_aggregator2d("attention", cc, cm)
    out = net((zc, zm))
    assert out.size() == (b, cc, h, w)


def test_attention_aggregator3d():
    b, cc, cm, d, h, w = 8, 4, 8, 2, 2, 2
    zc = randn((b, cc, d, h, w))
    zm = randn((b, cm, d, h, w))

    net = create_aggregator3d("attention", cc, cm)
    out = net((zc, zm))
    assert out.size() == (b, cc, d, h, w)
