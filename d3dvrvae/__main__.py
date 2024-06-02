import hydra
from omegaconf import OmegaConf, DictConfig

from .option import TrainExpOption, save_options, process_options
from .train import train


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    opt = OmegaConf.to_object(cfg)
    opt = process_options(opt)
    save_options(opt, opt.experiment.result_dir)
    if isinstance(opt.experiment, TrainExpOption):
        train(opt.experiment)
        return
    raise NotImplementedError(f'{opt.experiment.__class__.__name__} is not implemented')


main()
