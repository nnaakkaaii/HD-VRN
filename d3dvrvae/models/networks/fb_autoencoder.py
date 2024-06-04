from torch import nn

from .modules import ConvEncoder2d, ConvEncoder3d, ConvDecoder3d


class FiveBranchAutoencoder3d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 conv_params_2d: list[dict[str, int]],
                 conv_params_3d: list[dict[str, int]],
                 debug_show_dim: bool,
                 ) -> None:
        super().__init__()

        self.encoder_x_2d_ct = ConvEncoder2d(in_channels, latent_dim, conv_params_2d, debug_show_dim)
        self.encoder_x_exhale_2d_ct = ConvEncoder2d(in_channels, latent_dim, conv_params_2d, debug_show_dim)
        self.encoder_x_inhale_2d_ct = ConvEncoder2d(in_channels, latent_dim, conv_params_2d, debug_show_dim)
        self.encoder_x_exhale_3d_ct = ConvEncoder3d(in_channels, latent_dim, conv_params_3d, debug_show_dim)
        self.encoder_x_inhale_3d_ct = ConvEncoder3d(in_channels, latent_dim, conv_params_3d, debug_show_dim)
        self.deconv = ConvDecoder3d(in_channels, 5 * latent_dim, conv_params_3d, debug_show_dim)



