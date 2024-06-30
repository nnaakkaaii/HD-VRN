from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf

from .dataloaders import create_dataloader
from .models import create_model
from .option import Option, TrainExpOption, save_options


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    opt: Option = OmegaConf.to_object(cfg)  # type: ignore
    opt.experiment.result_dir = (
        opt.experiment.result_dir
        / opt.experiment.dataloader.__class__.__name__
        / opt.experiment.model.__class__.__name__
    )
    if hasattr(opt.experiment.model, "network"):
        opt.experiment.result_dir /= opt.experiment.model.network.__class__.__name__
    if hasattr(opt.experiment.model, "generator"):
        opt.experiment.result_dir /= opt.experiment.model.generator.__class__.__name__
    opt.experiment.result_dir /= datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_options(opt, opt.experiment.result_dir)
    if (
        isinstance(opt.experiment, TrainExpOption)
        and type(opt.experiment) is TrainExpOption
    ):
        train(opt.experiment)
        return
    raise NotImplementedError(f"{opt.experiment.__class__.__name__} is not implemented")


def train(opt: TrainExpOption) -> None:
    train_loader, val_loader = create_dataloader(opt.dataloader, is_train=True)
    model = create_model(opt.model, opt.n_epoch, steps_per_epoch=len(train_loader))

    model.train(
        train_loader,
        val_loader,
        n_epoch=opt.n_epoch,
        result_dir=opt.result_dir,
        debug=opt.debug,
    )


main()
