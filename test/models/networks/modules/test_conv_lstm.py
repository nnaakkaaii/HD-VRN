from torch import randn

from hrdae.models.networks.modules.conv_lstm import ConvLSTM1d, ConvLSTM2d


def test_conv_lstm1d():
    b, n, c, h = 8, 10, 16, 4
    latent, k, layer = 32, 3, 2
    x = randn((b, n, c, h))
    convlstm = ConvLSTM1d(c, latent, k, layer, True, True)
    y, last_states = convlstm(x)
    assert y.size() == (b, n, latent, h)
    assert len(last_states) == layer
    assert last_states[0][0].size() == (b, latent, h)


def test_conv_lstm2d():
    b, n, c, d, h = 8, 10, 16, 4, 4
    latent, k, layer = 32, 3, 2
    x = randn((b, n, c, d, h))
    convlstm = ConvLSTM2d(c, latent, (k, k), layer, True, True)
    y, last_states = convlstm(x)
    assert y.size() == (b, n, latent, d, h)
    assert len(last_states) == layer
    assert last_states[0][0].size() == (b, latent, d, h)
