import json
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import MISSING
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from .functions import save_reconstructed_images
from .losses import LossMixer, LossOption, create_loss
from .networks import NetworkOption, create_network
from .optimizers import OptimizerOption, create_optimizer
from .option import ModelOption
from .schedulers import LRScheduler, SchedulerOption, create_scheduler
from .typing import Model


@dataclass
class VRModelOption(ModelOption):
    network: NetworkOption = MISSING
    optimizer: OptimizerOption = MISSING
    scheduler: SchedulerOption = MISSING
    loss: dict[str, LossOption] = MISSING
    loss_coef: dict[str, float] = MISSING

    phase: str = "all"  # "all", "0", "t"
    pred_diff: bool = False


def save_model(model: nn.Module, filepath: Path):
    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
    torch.save(model_to_save.state_dict(), filepath)


class VRModel(Model):
    def __init__(
        self,
        network: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        criterion: nn.Module,
        phase: str = "all",  # "all", "0", "t"
        pred_diff: bool = False,
    ) -> None:
        self.network = network
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

        self.phase = phase
        self.pred_diff = pred_diff

        if torch.cuda.is_available():
            print("GPU is enabled")
            self.device = torch.device("cuda:0")
            self.network = nn.DataParallel(network).to(self.device)
        else:
            print("GPU is not enabled")
            self.device = torch.device("cpu")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epoch: int,
        result_dir: Path,
        debug: bool,
    ) -> float:
        max_iter = None
        if debug:
            max_iter = 5

        least_val_loss = float("inf")
        training_history: dict[str, list[dict[str, int | float]]] = {"history": []}

        for epoch in range(n_epoch):
            self.network.train()
            running_loss = 0.0

            for idx, data in enumerate(train_loader):
                if max_iter and max_iter <= idx:
                    break

                xm = data["x-"].to(self.device)
                xm_0 = data["x-_" + self.phase].to(self.device)
                xp = data["x+"].to(self.device)
                xp_0 = data["x+_" + self.phase].to(self.device)
                idx_expanded = data["idx_expanded"].to(self.device)

                self.optimizer.zero_grad()
                y = self.network(xm, xp_0, xm_0)

                if self.pred_diff:
                    assert self.phase != "all"
                    xp -= xp_0.unsqueeze(1)

                loss = self.criterion(y, xp, idx_expanded=idx_expanded)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if idx % 100 == 0:
                    print(f"Epoch: {epoch+1}, Batch: {idx} Loss: {loss.item():.6f}")

            running_loss /= len(train_loader)
            print(f"Epoch: {epoch+1}, Average Loss: {running_loss:.6f}")

            self.scheduler.step()

            self.network.eval()
            with torch.no_grad():
                total_val_loss = 0.0
                for idx, data in enumerate(val_loader):
                    if max_iter and max_iter <= idx:
                        break

                    xm = data["x-"].to(self.device)
                    xm_0 = data["x-_" + self.phase].to(self.device)
                    xp = data["x+"].to(self.device)
                    xp_0 = data["x+_" + self.phase].to(self.device)
                    idx_expanded = data["idx_expanded"].to(self.device)
                    y = self.network(xm, xp_0, xm_0)

                    if self.pred_diff:
                        assert self.phase != "all"
                        xp -= xp_0.unsqueeze(1)

                    loss = self.criterion(y, xp, idx_expanded=idx_expanded)
                    total_val_loss += loss.item()

                avg_val_loss = total_val_loss / len(val_loader)
                print(f"Epoch: {epoch+1}, Val Loss: {avg_val_loss:.6f}")

                if avg_val_loss < least_val_loss:
                    least_val_loss = avg_val_loss
                    save_model(self.network, result_dir / "best_model.pth")

            training_history["history"].append(
                {
                    "epoch": int(epoch + 1),
                    "train_loss": float(running_loss),
                    "val_loss": float(avg_val_loss),
                }
            )

            if epoch % 10 == 0:
                data = next(iter(val_loader))

                xm = data["x-"].to(self.device)
                xm_0 = data["x-_" + self.phase].to(self.device)
                xp = data["x+"].to(self.device)
                xp_0 = data["x+_" + self.phase].to(self.device)

                y = self.network(xm, xp_0, xm_0)

                if self.pred_diff:
                    assert self.phase != "all"
                    y += xp_0.unsqueeze(1)

                save_reconstructed_images(
                    xp.data.cpu().clone().detach().numpy(),
                    y.data.cpu().clone().detach().numpy(),
                    epoch,
                    result_dir / "logs" / "reconstructed",
                )
                save_model(self.network, result_dir / f"model_{epoch}.pth")

        with open(result_dir / "training_history.json", "w") as f:
            json.dump(training_history, f)

        return least_val_loss


def create_vr_model(
    opt: VRModelOption,
    n_epoch: int,
    steps_per_epoch: int,
) -> Model:
    in_channels = 2 if opt.phase == "all" else 1
    network = create_network(in_channels, 1, opt.network)
    optimizer = create_optimizer(
        opt.optimizer,
        network.parameters(),
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
    return VRModel(
        network,
        optimizer,
        scheduler,
        criterion,
        opt.phase,
        opt.pred_diff,
    )
