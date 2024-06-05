from dataclasses import dataclass

from torch import Tensor, cat, nn
from torch.nn.functional import mse_loss


@dataclass
class PJCLossOption:
    pass


def create_pjc_loss(opt: PJCLossOption) -> nn.Module:
    return PJCLoss()


class PJCLoss(nn.Module):
    def forward(
        self,
        reconstructed_3d: Tensor,
        input_2d: Tensor,
        slice_idx: Tensor,
    ) -> Tensor:
        sliced_tensors = []
        for i, idx in enumerate(slice_idx):
            sliced_tensor = reconstructed_3d[i, :, :, idx].unsqueeze(0)
            sliced_tensors.append(sliced_tensor)
        loss = mse_loss(cat(sliced_tensors, dim=0), input_2d)
        return loss
