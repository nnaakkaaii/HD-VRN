from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import MISSING, OmegaConf

from .dataloaders import DataLoaderOption
from .models import ModelOption


@dataclass
class ExpOption:
    result_dir: Path = field(default_factory=Path)
    debug: bool = False

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
    experiment: ExpOption = MISSING


def save_options(opt: Option, save_dir: Path) -> None:
    config = OmegaConf.structured(opt)
    save_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, save_dir / "options.yaml")
    return
