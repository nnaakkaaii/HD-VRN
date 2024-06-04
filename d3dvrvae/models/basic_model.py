from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from omegaconf import MISSING

from ..dataloaders import Data
from .losses import LossOption, create_loss
from .networks import NetworkOption, create_network
from .optimizers import OptimizerOption, create_optimizer
from .schedulers import SchedulerOption, create_scheduler, LRScheduler
from .option import ModelOption
from .typing import Model
from .functions import save_reconstructed_images


@dataclass
class BasicModelOption(ModelOption):
    network: NetworkOption = MISSING
    optimizer: OptimizerOption = MISSING
    scheduler: SchedulerOption = MISSING
    loss: LossOption = MISSING


class BasicModel(Model):
    def __init__(self,
                 network: nn.Module,
                 optimizer: Optimizer,
                 scheduler: LRScheduler,
                 criterion: nn.Module,
                 max_iter: int | None = None,
                 ) -> None:
        self.network = network
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.max_iter = max_iter

        if torch.cuda.is_available():
            print("GPU is enabled")
            self.device = torch.device("cuda:0")
        else:
            print("GPU is not enabled")
            self.device = torch.device("cpu")

    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              n_epoch: int,
              result_dir: str,
              ) -> None:
        x: Data
        y: Data
        t: Data
        self.network.to(self.device)

        for epoch in range(n_epoch):
            self.network.train()
            running_loss = 0.0

            for idx, (x, t) in enumerate(train_loader):
                if self.max_iter and self.max_iter <= idx:
                    break

                x = x.to(self.device)
                t = t.to(self.device)

                self.optimizer.zero_grad()
                y = self.network(x)
                loss = self.criterion(t, y)
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
                for idx, (x, t) in enumerate(val_loader):
                    if self.max_iter and self.max_iter <= idx:
                        break

                    x = x.to(self.device)
                    t = t.to(self.device)

                    y = self.network(x)
                    loss = self.criterion(t, y)
                    total_val_loss += loss.item()

                avg_val_loss = total_val_loss / len(val_loader)
                print(f"Epoch: {epoch+1}, Val Loss: {avg_val_loss:.6f}")

            if epoch % 10 == 0:
                x, t = next(iter(val_loader))

                x = x.to(self.device)
                t = t.to(self.device)

                y = self.network(x)

                save_reconstructed_images(
                    t.data.cpu().clone().detach().numpy()[:10],
                    y.data.cpu().clone().detach().numpy()[:10],
                    epoch,
                    result_dir / "logs" / "reconstructed",
                )


def create_basic_model(
        opt: BasicModelOption,
        n_epoch: int,
        steps_per_epoch: int,
) -> BasicModel:
    network = create_network(opt.network)
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
    criterion = create_loss(opt.loss)
    return BasicModel(network, optimizer, scheduler, criterion)
