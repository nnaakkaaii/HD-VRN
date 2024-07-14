from torch import randn

from hrdae.models.losses import TripletLossOption, create_loss


def test_triplet_loss():
    triplet_loss = create_loss(TripletLossOption())
    b, n, c, w, h = 8, 10, 8, 4, 4

    input = randn(b, n, w, h)
    target = randn(b, n, w, h)
    latent = randn(b, 1, c, w, h)
    positive = randn(b, n, c, w, h)
    negative = randn(b, n, c, w, h)

    loss = triplet_loss(
        input,
        target,
        latent=[latent],
        positive=[positive],
        negative=[negative],
    )
    assert loss.size() == ()
