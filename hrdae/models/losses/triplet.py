from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .option import LossOption


@dataclass
class TripletLossOption(LossOption):
    margin: float = 0.1


def create_triplet_loss(opt: TripletLossOption) -> nn.Module:
    return TripletLoss(opt.margin)


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0.1) -> None:
        super().__init__()
        self.margin = margin

    @property
    def required_kwargs(self) -> list[str]:
        return ["latent", "positive", "negative"]

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        latent: list[Tensor],
        positive: list[Tensor],
        negative: list[Tensor],
    ) -> Tensor:
        num_frames = positive[0].size(1)

        anchor = latent[0].unsqueeze(1).expand(-1, num_frames, -1, -1, -1)
        pos_dist = torch.norm(anchor - positive[0], p=2, dim=[2, 3, 4])
        neg_dist = torch.norm(anchor - negative[0], p=2, dim=[2, 3, 4])

        hard_positive_dist = pos_dist.max(dim=1)[0]
        hard_negative_dist = neg_dist.min(dim=1)[0]

        losses = F.relu(hard_positive_dist - hard_negative_dist + self.margin)
        return losses.mean()
