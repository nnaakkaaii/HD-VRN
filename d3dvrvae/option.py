from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import MISSING, OmegaConf

from .dataloaders import DataLoaderOption
from .models import ModelOption


@dataclass
class ExpOption:
    run_name: str = MISSING
    result_dir: Path = field(default_factory=Path)

    dataloader: DataLoaderOption = MISSING
    model: ModelOption = MISSING


@dataclass
class TrainExpOption(ExpOption):
    n_epoch: int = 50


@dataclass
class TestExpOption(ExpOption):
    pass


@dataclass
class Option:
    task_name: str = MISSING

    experiment: ExpOption = MISSING


def process_options(opt: Option) -> Option:
    opt.experiment.result_dir = (
        opt.experiment.result_dir / opt.task_name / opt.experiment.run_name
    )
    return opt


def save_options(opt: Option, save_dir: Path) -> None:
    config = OmegaConf.structured(opt)
    save_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, save_dir / "options.yaml")
    return
