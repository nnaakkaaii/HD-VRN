from torch import randn

from hrdae.models.networks.modules.tcn import TCN1d, TCN2d


def test_tcn1d():
    b, n, c, h = 8, 10, 16, 4
    latent, k, layer = 32, 3, 2
    x = randn((b, n, c, h))
    tcn1d = TCN1d(c, [latent] * layer, k, h)
    y = tcn1d(x)
    assert y.size() == (b, n, latent, h)


def test_tcn2d():
    b, n, c, d, h = 8, 10, 16, 4, 4
    latent, k, layer = 32, 3, 2
    x = randn((b, n, c, d, h))
    tcn2d = TCN2d(c, [latent] * layer, k, (d, h))
    y = tcn2d(x)
    assert y.size() == (b, n, latent, d, h)
