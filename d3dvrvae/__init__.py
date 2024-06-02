from hydra.core.config_store import ConfigStore

from .option import Option, TrainExpOption, TestExpOption
from .dataloaders import DataLoaderOption
from .datasets import MNISTDatasetOption
from .losses import MSELossOption
from .networks import AutoEncoder2dNetworkOption
from .optimizers import AdamOptimizerOption
from .schedulers import OneCycleLRSchedulerOption
from .transforms import BasicTransformOption


cs = ConfigStore.instance()
cs.store(name="config_schema", node=Option)
cs.store(group="config/experiment", name="train", node=TrainExpOption)
cs.store(group="config/experiment", name="test", node=TestExpOption)
cs.store(group="config/experiment/dataloader", name="defaults", node=DataLoaderOption)
cs.store(group="config/experiment/dataset", name="mnist", node=MNISTDatasetOption)
cs.store(group="config/experiment/loss", name="mse", node=MSELossOption)
cs.store(group="config/experiment/network", name="autoencoder2d", node=AutoEncoder2dNetworkOption)
cs.store(group="config/experiment/optimizer", name="adam", node=AdamOptimizerOption)
cs.store(group="config/experiment/scheduler", name="onecyclelr", node=OneCycleLRSchedulerOption)
cs.store(group="config/experiment/transform", name="basic", node=BasicTransformOption)
