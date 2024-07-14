import json
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import MISSING
from torch import nn, tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from .functions import save_model, save_reconstructed_images
from .losses import LossMixer, LossOption, create_loss
from .networks import NetworkOption, create_network
from .optimizers import OptimizerOption, create_optimizer
from .option import ModelOption
from .schedulers import LRScheduler, SchedulerOption, create_scheduler
from .typing import Model


@dataclass
class BasicModelOption(ModelOption):
    network: NetworkOption = MISSING
    network_weight: str = ""
    optimizer: OptimizerOption = MISSING
    scheduler: SchedulerOption = MISSING
    loss: dict[str, LossOption] = MISSING
    loss_coef: dict[str, float] = MISSING

    serialize: bool = False


class BasicModel(Model):
    def __init__(
        self,
        network: nn.Module,
        network_weight: str,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        criterion: nn.Module,
        serialize: bool = False,
    ) -> None:
        self.network = network
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.serialize = serialize

        if network_weight != "":
            self.network.load_state_dict(torch.load(network_weight))

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

        self.network.to(self.device)

        least_val_loss = float("inf")
        training_history: dict[str, list[dict[str, int | float]]] = {"history": []}

        for epoch in range(n_epoch):
            self.network.train()
            running_loss = 0.0

            for idx, data in enumerate(train_loader):
                if max_iter and max_iter <= idx:
                    break

                x = data["xp"].to(self.device)
                t = data["xp"].to(self.device)

                b, n = x.size()[:2]

                self.optimizer.zero_grad()
                if self.serialize:
                    x = x.reshape(b * n, *x.size()[2:])
                y, z = self.network(x)
                if self.serialize:
                    y = y.reshape(b, n, *y.size()[1:])
                    z = z.reshape(b, n, *z.size()[1:])

                loss = self.criterion(y, t, latent=z)
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
                t = tensor([0.0], device=self.device)
                y = tensor([0.0], device=self.device)
                for idx, data in enumerate(val_loader):
                    if max_iter and max_iter <= idx:
                        break

                    x = data["xp"].to(self.device)
                    t = data["xp"].to(self.device)

                    b, n = x.size()[:2]

                    if self.serialize:
                        x = x.reshape(b * n, *x.size()[2:])
                    y, z = self.network(x)
                    if self.serialize:
                        y = y.reshape(b, n, *y.size()[1:])
                        z = z.reshape(b, n, *z.size()[1:])

                    loss = self.criterion(y, t, latent=z)
                    total_val_loss += loss.item()

                avg_val_loss = total_val_loss / len(val_loader)
                print(f"Epoch: {epoch+1}, Val Loss: {avg_val_loss:.6f}")

                if avg_val_loss < least_val_loss:
                    least_val_loss = avg_val_loss
                    save_reconstructed_images(
                        t.data.cpu().clone().detach().numpy()[:10],
                        y.data.cpu().clone().detach().numpy()[:10],
                        "best",
                        result_dir / "logs" / "reconstructed",
                    )
                    _save_model(
                        self.network,
                        result_dir / "weights",
                        "best",
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

                x = data["xp"].to(self.device)
                t = data["xp"].to(self.device)

                b, n = x.size()[:2]

                if self.serialize:
                    x = x.reshape(b * n, *x.size()[2:])
                y, _ = self.network(x)
                if self.serialize:
                    y = y.reshape(b, n, *y.size()[1:])

                save_reconstructed_images(
                    t.data.cpu().clone().detach().numpy()[:10],
                    y.data.cpu().clone().detach().numpy()[:10],
                    f"epoch_{epoch}",
                    result_dir / "logs" / "reconstructed",
                )
                _save_model(
                    self.network,
                    result_dir / "weights",
                    f"epoch_{epoch}",
                )

        return least_val_loss


def _save_model(module: nn.Module, save_dir: Path, name: str) -> None:
    if isinstance(module, nn.DataParallel):
        module = module.module
    if hasattr(module, "encoder"):
        save_model(
            module.encoder,
            save_dir / f"{name}_encoder.pth",
        )
    if hasattr(module, "decoder"):
        save_model(
            module.decoder,
            save_dir / f"{name}_decoder.pth",
        )
    save_model(
        module,
        save_dir / f"{name}_model.pth",
    )


def create_basic_model(
    opt: BasicModelOption,
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
    return BasicModel(network, opt.network_weight, optimizer, scheduler, criterion, opt.serialize)
