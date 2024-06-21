# Pretrained Video Reconstruction Model
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import MISSING
from torch import nn
from torch.optim.optimizer import Optimizer

from .losses import LossMixer, create_loss
from .networks import create_network
from .optimizers import create_optimizer
from .schedulers import LRScheduler, create_scheduler
from .typing import Model
from .vr_model import VRModel, VRModelOption


@dataclass
class PVRModelOption(VRModelOption):
    network_weight: dict[str, Path] = MISSING
    network_grad: dict[str, bool] = MISSING


class PVRModel(VRModel):
    def __init__(
        self,
        network: nn.Module,
        network_weight: dict[str, Path],
        network_grad: dict[str, bool],
        optimizer: Optimizer,
        scheduler: LRScheduler,
        criterion: nn.Module,
        phase: str = "all",  # "all", "0", "t"
        pred_diff: bool = False,
    ) -> None:
        for k, v in network_weight.items():
            if not hasattr(network, k):
                raise ValueError(f"{k} is not an attribute of the network")
            getattr(network, k).load_state_dict(torch.load(v))
        for k, b in network_grad.items():
            if b:
                continue
            if not hasattr(network, k):
                raise ValueError(f"{k} is not an attribute of the network")
            for param in getattr(network, k).parameters():
                param.requires_grad = b

        super().__init__(
            network,
            optimizer,
            scheduler,
            criterion,
            phase,
            pred_diff,
        )


def create_pvr_model(
    opt: PVRModelOption,
    n_epoch: int,
    steps_per_epoch: int,
) -> Model:
    in_channels = 2 if opt.phase == "all" else 1
    network = create_network(in_channels, 1, opt.network)
    params = {}
    for k in ["content_encoder", "motion_encoder", "decoder"]:
        if hasattr(network, k):
            params[k] = getattr(network, k).parameters()
    optimizer = create_optimizer(
        opt.optimizer,
        params,
    )
    scheduler = create_scheduler(
        opt.scheduler,
        optimizer,
        n_epoch,
        steps_per_epoch,
    )
    criterion = LossMixer(
        {k: create_loss(v) for k, v in opt.loss.items()}, opt.loss_coef
    )
    return PVRModel(
        network,
        opt.network_weight,
        opt.network_grad,
        optimizer,
        scheduler,
        criterion,
        opt.phase,
        opt.pred_diff,
    )
