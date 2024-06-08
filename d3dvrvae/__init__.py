from hydra.core.config_store import ConfigStore

from .dataloaders import BasicDataLoaderOption
from .dataloaders.datasets import MNISTDatasetOption, CTDatasetOption
from .dataloaders.transforms import BasicTransform2dOption, BasicTransform3dOption
from .models import BasicModelOption
from .models.losses import MSELossOption
from .models.networks import AutoEncoder2dNetworkOption
from .models.optimizers import AdamOptimizerOption
from .models.schedulers import OneCycleLRSchedulerOption
from .option import Option, TestExpOption, TrainExpOption

cs = ConfigStore.instance()
cs.store(name="config_schema", node=Option)
cs.store(group="config/experiment", name="train", node=TrainExpOption)
cs.store(group="config/experiment", name="test", node=TestExpOption)
cs.store(group="config/experiment/dataloader", name="basic", node=BasicDataLoaderOption)
cs.store(
    group="config/experiment/dataloader/dataset", name="mnist", node=MNISTDatasetOption
)
cs.store(
    group="config/experiment/dataloader/ct", name="ct", node=CTDatasetOption,
)
cs.store(
    group="config/experiment/dataloader/transform",
    name="basic2d",
    node=BasicTransform2dOption,
)
cs.store(
    group="config/experiment/dataloader/transform",
    name="basic3d",
    node=BasicTransform3dOption,
)
cs.store(group="config/experiment/model", name="basic", node=BasicModelOption)
cs.store(group="config/experiment/model/loss", name="mse", node=MSELossOption)
cs.store(
    group="config/experiment/model/network",
    name="autoencoder2d",
    node=AutoEncoder2dNetworkOption,
)
cs.store(
    group="config/experiment/model/optimizer", name="adam", node=AdamOptimizerOption
)
cs.store(
    group="config/experiment/model/scheduler",
    name="onecyclelr",
    node=OneCycleLRSchedulerOption,
)
