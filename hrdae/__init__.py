from hydra.core.config_store import ConfigStore

from .dataloaders import BasicDataLoaderOption
from .dataloaders.datasets import (
    CTDatasetOption,
    MNISTDatasetOption,
    MovingMNISTDatasetOption,
)
from .dataloaders.transforms import (
    Crop2dOption,
    MinMaxNormalizationOption,
    Normalize2dOption,
    Pad2dOption,
    Pool3dOption,
    RandomShift3dOption,
    ToTensorOption,
    UniformShape3dOption,
)
from .models import BasicModelOption, VRModelOption, PVRModelOption, GANModelOption
from .models.losses import (
    MSELossOption,
    BCEWithLogitsLossOption,
    PJC2dLossOption,
    PJC3dLossOption,
    WeightedMSELossOption,
    TemporalSimilarityLossOption,
)
from .models.networks import (
    Discriminator2dOption,
    Discriminator3dOption,
    AutoEncoder2dNetworkOption,
    AutoEncoder3dNetworkOption,
    HRDAE2dOption,
    HRDAE3dOption,
    RAE2dOption,
    RAE3dOption,
    RDAE2dOption,
    RDAE3dOption,
)
from .models.networks.motion_encoder import (
    MotionConv2dEncoder1dOption,
    MotionConv3dEncoder2dOption,
    MotionGuidedEncoder1dOption,
    MotionGuidedEncoder2dOption,
    MotionNormalEncoder1dOption,
    MotionNormalEncoder2dOption,
    MotionRNNEncoder1dOption,
    MotionRNNEncoder2dOption,
)
from .models.networks.rnn import (
    ConvLSTM1dOption,
    ConvLSTM2dOption,
    GRU1dOption,
    GRU2dOption,
    TCN1dOption,
    TCN2dOption,
)
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
    group="config/experiment/dataloader/dataset",
    name="moving_mnist",
    node=MovingMNISTDatasetOption,
)
cs.store(
    group="config/experiment/dataloader/dataset",
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
cs.store(group="config/experiment/model", name="vr", node=VRModelOption)
cs.store(group="config/experiment/model", name="pvr", node=PVRModelOption)
cs.store(group="config/experiment/model", name="gan", node=GANModelOption)
cs.store(group="config/experiment/model/loss", name="mse", node=MSELossOption)
cs.store(
    group="config/experiment/model/loss_g", name="bce", node=BCEWithLogitsLossOption
)
cs.store(
    group="config/experiment/model/loss_d", name="bce", node=BCEWithLogitsLossOption
)
cs.store(group="config/experiment/model/loss", name="pjc2d", node=PJC2dLossOption)
cs.store(group="config/experiment/model/loss", name="pjc3d", node=PJC3dLossOption)
cs.store(group="config/experiment/model/loss", name="wmse", node=WeightedMSELossOption)
cs.store(
    group="config/experiment/model/loss", name="tsim", node=TemporalSimilarityLossOption
)
cs.store(
    group="config/experiment/model/loss_g",
    name="tsim",
    node=TemporalSimilarityLossOption,
)
cs.store(
    group="config/experiment/model/network",
    name="autoencoder2d",
    node=AutoEncoder2dNetworkOption,
)
cs.store(
    group="config/experiment/model/network",
    name="autoencoder3d",
    node=AutoEncoder3dNetworkOption,
)
cs.store(group="config/experiment/model/network", name="hrdae2d", node=HRDAE2dOption)
cs.store(group="config/experiment/model/network", name="hrdae3d", node=HRDAE3dOption)
cs.store(group="config/experiment/model/network", name="rae2d", node=RAE2dOption)
cs.store(group="config/experiment/model/network", name="rae3d", node=RAE3dOption)
cs.store(group="config/experiment/model/network", name="rdae2d", node=RDAE2dOption)
cs.store(group="config/experiment/model/network", name="rdae3d", node=RDAE3dOption)
cs.store(
    group="config/experiment/model/network/motion_encoder",
    name="rnn1d",
    node=MotionRNNEncoder1dOption,
)
cs.store(
    group="config/experiment/model/network/motion_encoder",
    name="rnn2d",
    node=MotionRNNEncoder2dOption,
)
cs.store(
    group="config/experiment/model/network/motion_encoder",
    name="normal1d",
    node=MotionNormalEncoder1dOption,
)
cs.store(
    group="config/experiment/model/network/motion_encoder",
    name="normal2d",
    node=MotionNormalEncoder2dOption,
)
cs.store(
    group="config/experiment/model/network/motion_encoder",
    name="conv2d",
    node=MotionConv2dEncoder1dOption,
)
cs.store(
    group="config/experiment/model/network/motion_encoder",
    name="conv3d",
    node=MotionConv3dEncoder2dOption,
)
cs.store(
    group="config/experiment/model/network/motion_encoder",
    name="guided1d",
    node=MotionGuidedEncoder1dOption,
)
cs.store(
    group="config/experiment/model/network/motion_encoder",
    name="guided2d",
    node=MotionGuidedEncoder2dOption,
)
cs.store(
    group="config/experiment/model/network/motion_encoder/rnn",
    name="conv_lstm1d",
    node=ConvLSTM1dOption,
)
cs.store(
    group="config/experiment/model/network/motion_encoder/rnn",
    name="conv_lstm2d",
    node=ConvLSTM2dOption,
)
cs.store(
    group="config/experiment/model/network/motion_encoder/rnn",
    name="gru1d",
    node=GRU1dOption,
)
cs.store(
    group="config/experiment/model/network/motion_encoder/rnn",
    name="gru2d",
    node=GRU2dOption,
)
cs.store(
    group="config/experiment/model/network/motion_encoder/rnn",
    name="tcn1d",
    node=TCN1dOption,
)
cs.store(
    group="config/experiment/model/network/motion_encoder/rnn",
    name="tcn2d",
    node=TCN2dOption,
)
cs.store(
    group="config/experiment/model/discriminator",
    name="discriminator2d",
    node=Discriminator2dOption,
)
cs.store(
    group="config/experiment/model/discriminator",
    name="discriminator3d",
    node=Discriminator3dOption,
)
cs.store(
    group="config/experiment/model/generator",
    name="autoencoder2d",
    node=AutoEncoder2dNetworkOption,
)
cs.store(
    group="config/experiment/model/generator",
    name="autoencoder3d",
    node=AutoEncoder3dNetworkOption,
)
cs.store(
    group="config/experiment/model/optimizer", name="adam", node=AdamOptimizerOption
)
cs.store(
    group="config/experiment/model/optimizer_g", name="adam", node=AdamOptimizerOption
)
cs.store(
    group="config/experiment/model/optimizer_d", name="adam", node=AdamOptimizerOption
)
cs.store(
    group="config/experiment/model/scheduler",
    name="onecyclelr",
    node=OneCycleLRSchedulerOption,
)
cs.store(
    group="config/experiment/model/scheduler_g",
    name="onecyclelr",
    node=OneCycleLRSchedulerOption,
)
cs.store(
    group="config/experiment/model/scheduler_d",
    name="onecyclelr",
    node=OneCycleLRSchedulerOption,
)
