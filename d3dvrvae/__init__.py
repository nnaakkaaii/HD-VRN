from hydra.core.config_store import ConfigStore

from .dataloaders import BasicDataLoaderOption
from .dataloaders.datasets import CTDatasetOption, MNISTDatasetOption
from .dataloaders.transforms import (Crop2dOption, MinMaxNormalizationOption,
                                     Normalize2dOption, Pad2dOption,
                                     Pool3dOption, RandomShift3dOption,
                                     ToTensorOption, UniformShape3dOption)
from .models import BasicModelOption
from .models.losses import MSELossOption, PJCLossOption, WeightedMSELossOption
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
    group="config/experiment/dataloader/ct",
    name="ct",
    node=CTDatasetOption,
)
cs.store(
    group="config/experiment/dataloader/transform",
    name="to_tensor",
    node=ToTensorOption,
)
cs.store(
    group="config/experiment/dataloader/transform",
    name="pad2d",
    node=Pad2dOption,
)
cs.store(
    group="config/experiment/dataloader/transform",
    name="normalize2d",
    node=Normalize2dOption,
)
cs.store(
    group="config/experiment/dataloader/transform",
    name="crop2d",
    node=Crop2dOption,
)
cs.store(
    group="config/experiment/dataloader/transform",
    name="min_max_normalization",
    node=MinMaxNormalizationOption,
)
cs.store(
    group="config/experiment/dataloader/transform",
    name="random_shift3d",
    node=RandomShift3dOption,
)
cs.store(
    group="config/experiment/dataloader/transform",
    name="uniform_shape3d",
    node=UniformShape3dOption,
)
cs.store(
    group="config/experiment/dataloader/transform",
    name="pool3d",
    node=Pool3dOption,
)
cs.store(group="config/experiment/model", name="basic", node=BasicModelOption)
cs.store(group="config/experiment/model/loss", name="mse", node=MSELossOption)
cs.store(group="config/experiment/model/loss", name="pjc", node=PJCLossOption)
cs.store(group="config/experiment/model/loss", name="wmse", node=WeightedMSELossOption)
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
