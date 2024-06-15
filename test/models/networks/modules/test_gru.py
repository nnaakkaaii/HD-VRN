from torch import randn

from hrdae.models.networks.modules.gru import GRU1d, GRU2d


def test_gru1d():
    b, n, c, h = 8, 10, 16, 4
    latent, layer = 32, 2
    x = randn((b, n, c, h))
    gru1d = GRU1d(c, latent, layer, h)
    y, last_states = gru1d(x)
    assert y.size() == (b, n, latent, h)
    assert last_states.size() == (layer, b, latent * h)


def test_gru2d():
    b, n, c, d, h = 8, 10, 16, 4, 4
    latent, layer = 32, 2
    x = randn((b, n, c, d, h))
    gru2d = GRU2d(c, latent, layer, [d, h])
    y, last_states = gru2d(x)
    assert y.size() == (b, n, latent, d, h)
    assert last_states.size() == (layer, b, latent * d * h)
