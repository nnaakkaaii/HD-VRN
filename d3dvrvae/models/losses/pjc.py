from dataclasses import dataclass

from torch import Tensor, cat, nn
from torch.nn.functional import mse_loss


@dataclass
class PJCLossOption:
    pass


def create_pjc_loss() -> nn.Module:
    return PJCLoss()


class PJCLoss(nn.Module):
    def forward(
        self,
        reconstructed_3d: Tensor,
        input_2d: Tensor,
        slice_idx: Tensor,
    ) -> Tensor:
        if len(slice_idx.shape) == 1:
            assert slice_idx.shape[0] == reconstructed_3d.shape[0]
            assert reconstructed_3d.shape[:-1] == input_2d.shape
            sliced_tensors = []
            for i, idx in enumerate(slice_idx):
                sliced_tensor = reconstructed_3d[i, :, :, idx].unsqueeze(0)
                sliced_tensors.append(sliced_tensor)
            loss = mse_loss(cat(sliced_tensors, dim=0), input_2d)
            return loss
        elif len(slice_idx.shape) == 2:
            assert slice_idx.shape[0] == reconstructed_3d.shape[0]
            assert slice_idx.shape[1] == input_2d.shape[-1]
            assert reconstructed_3d.shape[:-1] == input_2d.shape[:-1]

            sliced_tensors = []
            for i in range(slice_idx.shape[0]):
                for j in range(slice_idx.shape[1]):
                    idx = slice_idx[i, j]
                    sliced_tensor = reconstructed_3d[i, :, :, idx].unsqueeze(0).unsqueeze(-1)
                    sliced_tensors.append(sliced_tensor)
            sliced_3d = cat(sliced_tensors, dim=0)
            loss = mse_loss(sliced_3d, input_2d)
            return loss
