from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf, MISSING

from .datasets import DatasetOption
from .dataloaders import DataLoaderOption
from .losses import LossOption
from .networks import NetworkOption
from .optimizers import OptimizerOption
from .schedulers import SchedulerOption
from .transforms import TransformOption


@dataclass
class ExpOption:
    run_name: str = MISSING
    result_dir: Path = field(default_factory=Path)

    dataset: DatasetOption = MISSING
    dataloader: DataLoaderOption = MISSING
    transform: TransformOption = MISSING
    network: NetworkOption = MISSING


@dataclass
class TrainExpOption(ExpOption):
    n_epoch: int = 50
    train_val_ratio: float = 0.8
    debug: bool = False

    optimizer: OptimizerOption = MISSING
    scheduler: SchedulerOption = MISSING
    loss: LossOption = MISSING


@dataclass
class TestExpOption(ExpOption):
    pass


@dataclass
class Option:
    task_name: str = MISSING

    experiment: ExpOption = MISSING


def process_options(opt: Option) -> Option:
    opt.experiment.result_dir = opt.experiment.result_dir / opt.task_name / opt.experiment.run_name
    return opt


def save_options(opt: Option, save_dir: Path) -> None:
    config = OmegaConf.structured(opt)
    save_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, save_dir / 'options.yaml')
    return
