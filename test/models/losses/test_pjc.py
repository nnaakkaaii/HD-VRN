from torch import randn, randint

from hrdae.models.losses.pjc import PJC2dLoss, PJC3dLoss


def test_pjc2d_loss():
    b, n, c, h, w, s = 8, 10, 1, 16, 16, 3

    pjc2d_loss = PJC2dLoss()
    # reconstructed_2d: (b, n, c, h, w)
    # input_2d: (b, n, c, h, w)
    # slice_idx: (b, n, s, h)
    reconstructed_2d = randn(b, n, c, h, w)
    input_2d = randn(b, n, c, h, w)
    slice_idx = randint(0, w, (b, s)).unsqueeze(1).unsqueeze(3).repeat(1, n, 1, h)
    loss = pjc2d_loss(reconstructed_2d, input_2d, slice_idx)
    assert loss.size() == ()


def test_pjc2d_loss__value():
    b, n, c, h, w, s = 8, 10, 1, 16, 16, 3

    pjc2d_loss = PJC2dLoss()
    # input_2d: (b, n, c, h, w)
    # slice_idx: (b, n, s, h)
    input_2d = randn(b, n, c, h, w)
    slice_idx = randint(0, w, (b, s)).unsqueeze(1).unsqueeze(3).repeat(1, n, 1, h)
    loss = pjc2d_loss(input_2d, input_2d, slice_idx)
    assert loss == 0.0


def test_pjc3d_loss():
    b, n, c, d, h, w, s = 32, 10, 1, 16, 16, 16, 3

    pjc3d_loss = PJC3dLoss()
    # reconstructed_3d: (b, n, c, d, h, w)
    # input_3d: (b, n, c, d, h, w)
    # slice_idx: (b, n, s, d, h)
    reconstructed_3d = randn(b, n, c, d, h, w)
    input_3d = randn(b, n, c, d, h, w)
    slice_idx = (
        randint(0, w, (b, s))
        .unsqueeze(1)
        .unsqueeze(3)
        .unsqueeze(4)
        .repeat(1, n, 1, d, h)
    )
    loss = pjc3d_loss(reconstructed_3d, input_3d, slice_idx)
    assert loss.size() == ()


def test_pjc3d_loss__value():
    b, n, c, d, h, w, s = 32, 10, 1, 16, 16, 16, 3

    pjc3d_loss = PJC3dLoss()
    # input_3d: (b, n, c, d, h, w)
    # slice_idx: (b, n, s, d, h)
    input_3d = randn(b, n, c, d, h, w)
    slice_idx = (
        randint(0, w, (b, s))
        .unsqueeze(1)
        .unsqueeze(3)
        .unsqueeze(4)
        .repeat(1, n, 1, d, h)
    )
    loss = pjc3d_loss(input_3d, input_3d, slice_idx)
    assert loss == 0.0
