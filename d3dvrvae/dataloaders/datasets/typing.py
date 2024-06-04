from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class Data:
    data: Tensor
    latent: Tensor | None = field(default=None)

    def to(self, device: torch.device) -> 'Data':
        self.data = self.data.to(device)
        if self.latent:
            self.latent = self.latent.to(device)
        return self
