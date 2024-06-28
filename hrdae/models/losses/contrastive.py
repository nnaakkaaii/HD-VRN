from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .option import LossOption


@dataclass
class ContrastiveLossOption(LossOption):
    margin: float = 0.1


def create_contrastive_loss(opt: ContrastiveLossOption) -> nn.Module:
    return MStdLoss(opt.margin)


class MStdLoss(nn.Module):
    def __init__(self, margin: float = 0.1) -> None:
        super().__init__()
        self.margin = margin

    @property
    def required_kwargs(self) -> list[str]:
        return ["latent"]

    def forward(self, input: Tensor, target: Tensor, latent: list[Tensor]) -> Tensor:
        feature = latent[0]
        b, t = feature.size()[:2]
        feature = feature.view(b * t, -1)
        square_distances = torch.cdist(feature, feature, p=2)

        labels = 1 - torch.eye(b * t).to(input.device)
        for i in range(b):
            labels[i * t : (i + 1) * t, i * t : (i + 1) * t] = 0

        positive_loss = (1 - labels) * 0.5 * torch.pow(square_distances, 2)
        negative_loss = (
            labels
            * 0.5
            * torch.pow(torch.clamp(self.margin - square_distances, min=0.0), 2)
        )

        loss = torch.sum(positive_loss + negative_loss) / (b * t * (b * t - 1))

        return loss
