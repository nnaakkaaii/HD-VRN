import json
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import MISSING
from torch import nn, tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from .functions import save_model, save_reconstructed_images, shuffled_indices
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
    use_triplet: bool = False


class VRModel(Model):
    def __init__(
        self,
        network: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        criterion: nn.Module,
        use_triplet: bool,
    ) -> None:
        self.network = network
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.use_triplet = use_triplet

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

                xm = data["xm"].to(self.device)
                xm_0 = data["xm_0"].to(self.device)
                xp = data["xp"].to(self.device)
                xp_0 = data["xp_0"].to(self.device)

                self.optimizer.zero_grad()
                y, latent, cycled_latent = self.network(xm, xp_0, xm_0)
                # triplet
                indices = shuffled_indices(len(xp))
                positive = tensor(0.0)
                negative = tensor(0.0)
                if self.use_triplet:
                    _, _, positive = self.network(xm[indices], xp_0, xm_0[indices])
                    _, _, negative = self.network(xm, xp_0[indices], xm_0)

                loss = self.criterion(
                    y,
                    xp,
                    latent=latent,
                    cycled_latent=cycled_latent,
                    positive=positive,
                    negative=negative,
                )
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
                xp = tensor([0.0], device=self.device)
                y = tensor([0.0], device=self.device)

                for idx, data in enumerate(val_loader):
                    if max_iter and max_iter <= idx:
                        break

                    xm = data["xm"].to(self.device)
                    xm_0 = data["xm_0"].to(self.device)
                    xp = data["xp"].to(self.device)
                    xp_0 = data["xp_0"].to(self.device)
                    y, latent, cycled_latent = self.network(xm, xp_0, xm_0)
                    # triplet
                    indices = shuffled_indices(len(xp))
                    positive = tensor(0.0)
                    negative = tensor(0.0)
                    if self.use_triplet:
                        _, positive, _ = self.network(xm[indices], xp_0, xm_0[indices])
                        _, negative, _ = self.network(xm, xp_0[indices], xm_0)

                    loss = self.criterion(
                        y,
                        xp,
                        latent=latent,
                        cycled_latent=cycled_latent,
                        positive=positive,
                        negative=negative,
                    )
                    total_val_loss += loss.item()

                avg_val_loss = total_val_loss / len(val_loader)
                print(f"Epoch: {epoch+1}, Val Loss: {avg_val_loss:.6f}")

                if avg_val_loss < least_val_loss:
                    least_val_loss = avg_val_loss
                    save_reconstructed_images(
                        xp.data.cpu().clone().detach().numpy(),
                        y.data.cpu().clone().detach().numpy(),
                        "best",
                        result_dir / "logs" / "reconstructed",
                    )
                    save_model(
                        self.network,
                        result_dir / "weights" / "best_model.pth",
                    )

            training_history["history"].append(
                {
                    "epoch": int(epoch + 1),
                    "train_loss": float(running_loss),
                    "val_loss": float(avg_val_loss),
                }
            )
            with open(result_dir / "training_history.json", "w") as f:
                json.dump(training_history, f)

            if epoch % 10 == 0:
                data = next(iter(val_loader))

                xm = data["xm"].to(self.device)
                xm_0 = data["xm_0"].to(self.device)
                xp = data["xp"].to(self.device)
                xp_0 = data["xp_0"].to(self.device)

                y, _, _ = self.network(xm, xp_0, xm_0)

                save_reconstructed_images(
                    xp.data.cpu().clone().detach().numpy(),
                    y.data.cpu().clone().detach().numpy(),
                    f"epoch_{epoch}",
                    result_dir / "logs" / "reconstructed",
                )
                save_model(
                    self.network,
                    result_dir / "weights" / f"model_{epoch}.pth",
                )

        return least_val_loss


def create_vr_model(
    opt: VRModelOption,
    n_epoch: int,
    steps_per_epoch: int,
) -> Model:
    network = create_network(1, opt.network)
    optimizer = create_optimizer(
        opt.optimizer,
        {"default": network.parameters()},
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
        opt.use_triplet,
    )
