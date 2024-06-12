from dataclasses import dataclass, field

from torch import Tensor, cat, nn

from .modules import ConvDecoder3d, ConvEncoder2d, ConvEncoder3d


@dataclass
class FiveBranchAutoencoder3dOption:
    in_channels: int = 1
    latent_dim: int = 64
    conv_params_2d: list[dict[str, int]] = field(
        default_factory=lambda: [
            {"kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
            {"kernel_size": 3, "stride": 1, "padding": 1, "output_padding": 0},
        ]
    )
    conv_params_3d: list[dict[str, int]] = field(
        default_factory=lambda: [
            {"kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
            {"kernel_size": 3, "stride": 1, "padding": 1, "output_padding": 0},
        ]
    )
    weighted: bool = False
    debug_show_dim: bool = False


def create_fb_autoencoder3d(
    opt: FiveBranchAutoencoder3dOption,
) -> "FiveBranchAutoencoder3d":
    return FiveBranchAutoencoder3d(
        in_channels=opt.in_channels,
        latent_dim=opt.latent_dim,
        conv_params_2d=opt.conv_params_2d,
        conv_params_3d=opt.conv_params_3d,
        debug_show_dim=opt.debug_show_dim,
    )


class FiveBranchAutoencoder3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        conv_params_2d: list[dict[str, int]],
        conv_params_3d: list[dict[str, int]],
        debug_show_dim: bool,
    ) -> None:
        super().__init__()

        self.encoder_x_2d_ct = ConvEncoder2d(
            in_channels, latent_dim, conv_params_2d, debug_show_dim
        )
        self.encoder_x_exhale_2d_ct = ConvEncoder2d(
            in_channels, latent_dim, conv_params_2d, debug_show_dim
        )
        self.encoder_x_inhale_2d_ct = ConvEncoder2d(
            in_channels, latent_dim, conv_params_2d, debug_show_dim
        )
        self.encoder_x_exhale_3d_ct = ConvEncoder3d(
            in_channels, latent_dim, conv_params_3d, debug_show_dim
        )
        self.encoder_x_inhale_3d_ct = ConvEncoder3d(
            in_channels, latent_dim, conv_params_3d, debug_show_dim
        )
        self.deconv = ConvDecoder3d(
            in_channels, 5 * latent_dim, conv_params_3d, debug_show_dim
        )

    def forward(
        self,
        x_2d_ct: Tensor,
        exhale_3d_ct: Tensor,
        inhale_3d_ct: Tensor,
        exhale_2d_ct: Tensor,
        inhale_2d_ct: Tensor,
    ) -> tuple[Tensor, Tensor]:
        # encode
        exhale_3d_ct = self.encoder_exhale_3d_ct(exhale_3d_ct)
        inhale_3d_ct = self.encoder_inhale_3d_ct(inhale_3d_ct)
        w_dim = exhale_3d_ct.shape[-1]
        x_2d_ct = self.encoder_x_2d_ct(x_2d_ct).unsqueeze(4).repeat(1, 1, 1, 1, w_dim)
        exhale_2d_ct = (
            self.encoder_exhale_2d_ct(exhale_2d_ct)
            .unsqueeze(4)
            .repeat(1, 1, 1, 1, w_dim)
        )
        inhale_2d_ct = (
            self.encoder_inhale_2d_ct(inhale_2d_ct)
            .unsqueeze(4)
            .repeat(1, 1, 1, 1, w_dim)
        )

        # aggregate
        latent = cat(
            [
                x_2d_ct,
                exhale_2d_ct,
                inhale_2d_ct,
                exhale_3d_ct,
                inhale_3d_ct,
            ],
            dim=1,
        )

        # decode
        out = self.deconv(latent)
        return out, latent
