from torch import randn

from hrdae.models.networks.fb_autoencoder import (
    FiveBranchAutoencoder2d,
    FiveBranchAutoencoder3d,
)


def test_five_branch_autoencoder2d():
    b, n, c, s, h, w = 8, 10, 1, 3, 16, 16
    latent = 32

    x_1d = randn(b, n, s, h)
    x_2d_0 = randn(b, 2, c, h, w)
    x_1d_0 = randn(b, 2, s, h)

    model = FiveBranchAutoencoder2d(
        in_channels_1d=s,
        in_channels_2d=1,
        latent_dim=latent,
        conv_params_1d=[
            {
                "kernel_size": [3],
                "stride": [2],
                "padding": [1],
                "output_padding": [1],
            },
        ]
        * 2,
        conv_params_2d=[
            {
                "kernel_size": [3],
                "stride": [2],
                "padding": [1],
                "output_padding": [1],
            },
        ]
        * 2,
        debug_show_dim=True,
    )
    out = model(x_1d, x_2d_0, x_1d_0)
    assert out.size() == (b, n, c, h, w)


def test_five_branch_autoencoder3d():
    b, n, c, s, d, h, w = 8, 10, 1, 3, 16, 16, 16
    latent = 64

    x_2d = randn(b, n, s, d, h)
    x_3d_0 = randn(b, 2, c, d, h, w)
    x_2d_0 = randn(b, 2, s, d, h)

    model = FiveBranchAutoencoder3d(
        in_channels_2d=s,
        in_channels_3d=1,
        latent_dim=latent,
        conv_params_2d=[
            {
                "kernel_size": [3],
                "stride": [2],
                "padding": [1],
                "output_padding": [1],
            },
        ]
        * 2,
        conv_params_3d=[
            {
                "kernel_size": [3],
                "stride": [2],
                "padding": [1],
                "output_padding": [1],
            },
        ]
        * 2,
        debug_show_dim=True,
    )
    out = model(x_2d, x_3d_0, x_2d_0)
    assert out.size() == (b, n, c, d, h, w)
