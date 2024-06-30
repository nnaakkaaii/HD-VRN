import json
from dataclasses import dataclass
from pathlib import Path

import torch
from omegaconf import MISSING
from torch import nn
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
class GANModelOption(ModelOption):
    generator: NetworkOption = MISSING
    discriminator: NetworkOption = MISSING
    optimizer_g: OptimizerOption = MISSING
    optimizer_d: OptimizerOption = MISSING
    scheduler_g: SchedulerOption = MISSING
    scheduler_d: SchedulerOption = MISSING
    loss: dict[str, LossOption] = MISSING
    loss_coef: dict[str, float] = MISSING
    loss_g: LossOption = MISSING
    loss_d: LossOption = MISSING
    serialize: bool = False


class GANModel(Model):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        optimizer_g: Optimizer,
        optimizer_d: Optimizer,
        scheduler_g: LRScheduler,
        scheduler_d: LRScheduler,
        criterion: nn.Module,
        criterion_g: nn.Module,
        criterion_d: nn.Module,
        serialize: bool = False,
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.scheduler_g = scheduler_g
        self.scheduler_d = scheduler_d
        self.criterion = criterion
        self.criterion_g = criterion_g
        self.criterion_d = criterion_d
        self.serialize = serialize

        if torch.cuda.is_available():
            print("GPU is enabled")
            self.device = torch.device("cuda:0")
            self.generator = nn.DataParallel(generator).to(self.device)
            self.discriminator = nn.DataParallel(discriminator).to(self.device)
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

        least_val_g_loss = float("inf")
        training_history: dict[str, list[dict[str, int | float]]] = {"history": []}

        for epoch in range(n_epoch):
            self.generator.train()
            self.discriminator.train()

            running_g_loss = 0.0
            running_d_loss = 0.0

            for idx, data in enumerate(train_loader):
                if max_iter is not None and idx >= max_iter:
                    break

                xm = data["xm"].to(self.device)
                xm_0 = data["xm_0"].to(self.device)
                xp = data["xp"].to(self.device)
                xp_0 = data["xp_0"].to(self.device)

                # train generator
                self.optimizer_g.zero_grad()
                y, latent = self.generator(xm, xp_0, xm_0)

                y_pred = self.discriminator(y, xp)
                loss_g = self.criterion(y, xp, latent=latent) + self.criterion_g(
                    y_pred, torch.ones_like(y_pred)
                )
                loss_g.backward()
                self.optimizer_g.step()

                running_g_loss += loss_g.item()

                # train discriminator
                self.optimizer_d.zero_grad()
                xp_pred = self.discriminator(xp, xp)
                loss_d = self.criterion_d(
                    xp_pred, torch.ones_like(xp_pred)
                ) + self.criterion_d(y_pred.detach(), torch.zeros_like(y_pred))
                loss_d.backward()
                self.optimizer_d.step()

                running_d_loss += loss_d.item()

                if idx % 100 == 0:
                    print(
                        f"Epoch: {epoch+1}, "
                        f"Batch: {idx}, "
                        f"Loss D: {loss_d.item():.6f}, "
                        f"Loss G: {loss_g.item():.6f}"
                    )

            running_g_loss /= len(train_loader)
            running_d_loss /= len(train_loader)

            self.scheduler_g.step()
            self.scheduler_d.step()

            self.generator.eval()
            self.discriminator.eval()
            with torch.no_grad():
                total_val_g_loss = 0.0
                total_val_d_loss = 0.0
                xp = torch.tensor([0.0], device=self.device)
                y = torch.tensor([0.0], device=self.device)

                for idx, data in enumerate(val_loader):
                    if max_iter is not None and idx >= max_iter:
                        break

                    xm = data["xm"].to(self.device)
                    xm_0 = data["xm_0"].to(self.device)
                    xp = data["xp"].to(self.device)
                    xp_0 = data["xp_0"].to(self.device)
                    y, latent = self.generator(xm, xp_0, xm_0)

                    y_pred = self.discriminator(y, xp)
                    xp_pred = self.discriminator(xp, xp)
                    y = y.detach().clone()
                    loss_g = self.criterion(y, xp, latent=latent) + self.criterion_g(
                        y_pred, torch.ones_like(y_pred)
                    )
                    loss_d = self.criterion_d(
                        xp_pred, torch.ones_like(xp_pred)
                    ) + self.criterion_d(y_pred, torch.zeros_like(y_pred))

                    total_val_g_loss += loss_g.item()
                    total_val_d_loss += loss_d.item()

                total_val_g_loss /= len(val_loader)
                total_val_d_loss /= len(val_loader)

                print(
                    f"Epoch: {epoch+1} "
                    f"[train] "
                    f"Loss D: {running_d_loss:.6f}, "
                    f"Loss G: {running_g_loss:.6f} "
                    f"[val] "
                    f"Loss D: {total_val_d_loss:.6f}, "
                    f"Loss G: {total_val_g_loss:.6f}"
                )

                if total_val_g_loss < least_val_g_loss:
                    least_val_g_loss = total_val_g_loss
                    torch.save(
                        self.generator.state_dict(), result_dir / "generator.pth"
                    )
                    torch.save(
                        self.discriminator.state_dict(),
                        result_dir / "discriminator.pth",
                    )
                    save_reconstructed_images(
                        xp.data.cpu().clone().detach().numpy()[:10],
                        y.data.cpu().clone().detach().numpy()[:10],
                        "best",
                        result_dir / "logs" / "reconstructed",
                    )
                    _save_model(
                        self.generator,
                        self.discriminator,
                        result_dir / "weights",
                        "best",
                    )

            training_history["history"].append(
                {
                    "epoch": int(epoch + 1),
                    "train_loss_g": float(running_g_loss),
                    "train_loss_d": float(running_d_loss),
                    "val_loss_g": float(total_val_g_loss),
                    "val_loss_d": float(total_val_d_loss),
                }
            )

            if epoch % 10 == 0:
                data = next(iter(val_loader))

                xm = data["xm"].to(self.device)
                xm_0 = data["xm_0"].to(self.device)
                xp = data["xp"].to(self.device)
                xp_0 = data["xp_0"].to(self.device)

                y, _ = self.generator(xm, xp_0, xm_0)

                save_reconstructed_images(
                    xp.data.cpu().clone().detach().numpy()[:10],
                    y.data.cpu().clone().detach().numpy()[:10],
                    f"epoch_{epoch}",
                    result_dir / "logs" / "reconstructed",
                )
                _save_model(
                    self.generator,
                    self.discriminator,
                    result_dir / "weights",
                    f"epoch_{epoch}",
                )

        with open(result_dir / "training_history.json", "w") as f:
            json.dump(training_history, f)

        return least_val_g_loss


def _save_model(
    generator: nn.Module, discriminator: nn.Module, save_dir: Path, name: str
) -> None:
    if isinstance(generator, nn.DataParallel):
        generator = generator.module
    if isinstance(discriminator, nn.DataParallel):
        discriminator = discriminator.module
    if hasattr(generator, "encoder"):
        save_model(
            generator.encoder,
            save_dir / f"{name}_generator_encoder.pth",
        )
    if hasattr(generator, "decoder"):
        save_model(
            generator.encoder,
            save_dir / f"{name}_generator_decoder.pth",
        )
    save_model(
        generator,
        save_dir / f"{name}_generator.pth",
    )
    save_model(
        discriminator,
        save_dir / f"{name}_discriminator.pth",
    )


def create_gan_model(
    opt: GANModelOption,
    n_epoch: int,
    steps_per_epoch: int,
) -> Model:
    generator = create_network(1, opt.generator)
    discriminator = create_network(2, opt.discriminator)
    optimizer_g = create_optimizer(
        opt.optimizer_g,
        {"default": generator.parameters()},
    )
    optimizer_d = create_optimizer(
        opt.optimizer_d,
        {"default": discriminator.parameters()},
    )
    scheduler_g = create_scheduler(
        opt.scheduler_g,
        optimizer_g,
        n_epoch,
        steps_per_epoch,
    )
    scheduler_d = create_scheduler(
        opt.scheduler_d,
        optimizer_d,
        n_epoch,
        steps_per_epoch,
    )
    criterion = LossMixer(
        {k: create_loss(v) for k, v in opt.loss.items()},
        opt.loss_coef,
    )
    criterion_g = create_loss(opt.loss_g)
    criterion_d = create_loss(opt.loss_d)
    return GANModel(
        generator,
        discriminator,
        optimizer_g,
        optimizer_d,
        scheduler_g,
        scheduler_d,
        criterion,
        criterion_g,
        criterion_d,
        opt.serialize,
    )
