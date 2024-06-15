from torch import randn

from hrdae.models.networks.rnn import ConvLSTM1d, ConvLSTM2d, GRU1d, GRU2d, TCN1d, TCN2d


def test_conv_lstm1d():
    b, n, h = 8, 10, 4
    latent, layer = 32, 2
    x = randn((b, n, latent, h))
    convlstm = ConvLSTM1d(latent, layer)
    y, last_states = convlstm(x)
    assert y.size() == (b, n, latent, h)
    assert len(last_states) == layer
    assert last_states[0][0].size() == (b, latent, h)


def test_conv_lstm2d():
    b, n, d, h = 8, 10, 4, 4
    latent, layer = 32, 2
    x = randn((b, n, latent, d, h))
    convlstm = ConvLSTM2d(latent, layer)
    y, last_states = convlstm(x)
    assert y.size() == (b, n, latent, d, h)
    assert len(last_states) == layer
    assert last_states[0][0].size() == (b, latent, d, h)


def test_gru1d():
    b, n, h = 8, 10, 4
    latent, layer = 32, 2
    x = randn((b, n, latent, h))
    gru1d = GRU1d(latent, layer, h)
    y, last_states = gru1d(x)
    assert y.size() == (b, n, latent, h)
    assert last_states.size() == (layer, b, latent * h)


def test_gru2d():
    b, n, d, h = 8, 10, 4, 4
    latent, layer = 32, 2
    x = randn((b, n, latent, d, h))
    gru2d = GRU2d(latent, layer, [d, h])
    y, last_states = gru2d(x)
    assert y.size() == (b, n, latent, d, h)
    assert last_states.size() == (layer, b, latent * d * h)


def test_tcn1d():
    b, n, h = 8, 10, 4
    latent, k, layer = 32, 3, 2
    x = randn((b, n, latent, h))
    tcn1d = TCN1d(latent, layer, h, k)
    y, _ = tcn1d(x)
    assert y.size() == (b, n, latent, h)


def test_tcn2d():
    b, n, d, h = 8, 10, 4, 4
    latent, k, layer = 32, 3, 2
    x = randn((b, n, latent, d, h))
    tcn2d = TCN2d(latent, layer, (d, h), k)
    y, _ = tcn2d(x)
    assert y.size() == (b, n, latent, d, h)
