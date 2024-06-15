from torch import randn

from hrdae.models.losses.weighted_mse import WeightedMSELoss


def test_weighted_mse_loss__2d():
    wmse = WeightedMSELoss(weight_dynamic=0.5)
    b, n, w, h = 8, 10, 16, 16

    input = randn(b, n, w, h)
    target = randn(b, n, w, h)

    loss = wmse(input, target)
    assert loss.size() == ()


def test_weighted_mse_loss__2d_value():
    wmse = WeightedMSELoss(weight_dynamic=0.5)
    b, n, w, h = 8, 10, 16, 16

    target = randn(b, n, w, h)

    loss = wmse(target, target)
    assert loss == 0.0


def test_weighted_mse_loss__3d():
    wmse = WeightedMSELoss(weight_dynamic=0.5)
    b, n, d, w, h = 8, 10, 16, 16, 16

    input = randn(b, n, d, w, h)
    target = randn(b, n, d, w, h)

    loss = wmse(input, target)
    assert loss.size() == ()


def test_weighted_mse_loss__3d_value():
    wmse = WeightedMSELoss(weight_dynamic=0.5)
    b, n, d, w, h = 8, 10, 16, 16, 16

    target = randn(b, n, d, w, h)

    loss = wmse(target, target)
    assert loss == 0.0
