import json
from dataclasses import asdict
from pathlib import Path

from hrdae.option import TrainExpOption
from hrdae.dataloaders.transforms import MinMaxNormalizationOption
from hrdae.dataloaders.datasets import MovingMNISTDatasetOption
from hrdae.dataloaders import create_dataloader, BasicDataLoaderOption
from hrdae.models import create_model, VRModelOption
from hrdae.models.losses import WeightedMSELossOption
from hrdae.models.optimizers import AdamOptimizerOption
from hrdae.models.schedulers import OneCycleLRSchedulerOption
from hrdae.models.networks import RAE2dOption, RDAE2dOption, HRDAE2dOption
from hrdae.models.networks.rnn import ConvLSTM1dOption, GRU1dOption, TCN1dOption
from hrdae.models.networks.motion_encoder import MotionRNNEncoder1dOption, MotionConv2dEncoder1dOption, MotionGuidedEncoder1dOption, MotionNormalEncoder1dOption, MotionTSNEncoder1dOption


def interleave_arrays(
    an: list[dict[str, list[int]]],
    am: list[dict[str, list[int]]],
) -> list[dict[str, list[int]]]:
    result = []
    slots = len(an) - 1
    per_slot = len(am) // slots if slots > 0 else len(am)
    extra = len(am) % slots if slots > 0 else 0
    am_index = 0
    for i in range(len(an)):
        result.append(an[i])
        if i < len(an) - 1:
            num_elements = per_slot + (1 if extra > 0 else 0)
            result.extend(am[am_index:am_index + num_elements])
            am_index += num_elements
            if extra > 0:
                extra -= 1
    return result


def objective(trial):
    dataset_option = MovingMNISTDatasetOption(
        root="data",
        slice_index=[8, 16, 24, 32, 40, 48, 56],
    )

    transform_option = {
        "min_max_normalization": MinMaxNormalizationOption(),
    }

    dataloader_option = BasicDataLoaderOption(
        batch_size=96,
        train_val_ratio=0.8,
        dataset=dataset_option,
        transform_order_train=["min_max_normalization"],
        transform_order_val=["min_max_normalization"],
        transform=transform_option,
    )

    loss_option = {
        "wmse": WeightedMSELossOption(
            weight_dynamic=1.0,
        ),
    }

    phase = trial.suggest_categorical("phase", ["0", "t", "all"])
    network_name = trial.suggest_categorical("network", ["hrdae2d", "rae2d", "rdae2d"])
    if phase == "all":
        motion_encoder_name = trial.suggest_categorical("motion_encoder", ["conv2d", "normal1d", "rnn1d"])
    else:
        motion_encoder_name = trial.suggest_categorical("motion_encoder", ["conv2d", "guided1d", "normal1d", "rnn1d", "tsn1d"])
    motion_encoder_num_layers = trial.suggest_int("motion_encoder_num_layers", 0, 3)
    if motion_encoder_name == "rnn1d":
        rnn_name = trial.suggest_categorical("rnn", ["conv_lstm1d", "gru1d", "tcn1d"])
        motion_encoder_name = f"{motion_encoder_name}/{rnn_name}"
        if rnn_name == "conv_lstm1d":
            rnn_option = ConvLSTM1dOption(
                num_layers=trial.suggest_int("rnn_num_layers", 2, 4),
            )
        elif rnn_name == "gru1d":
            rnn_option = GRU1dOption(
                num_layers=trial.suggest_int("rnn_num_layers", 2, 4),
                image_size=8,
            )
        elif rnn_name == "tcn1d":
            rnn_option = TCN1dOption(
                num_layers=trial.suggest_int("rnn_num_layers", 2, 4),
                image_size=8,
                kernel_size=3,
                dropout=0.1,
            )
        else:
            raise RuntimeError("unreachable")
        motion_encoder_option = MotionRNNEncoder1dOption(
            in_channels=7,
            conv_params=interleave_arrays(
                [{"kernel_size": [3], "stride": [2], "padding": [1]}] * 3,
                [{"kernel_size": [3], "stride": [1], "padding": [1]}] * motion_encoder_num_layers,
            ),
            rnn=rnn_option,
        )
    elif motion_encoder_name == "conv2d":
        motion_encoder_option = MotionConv2dEncoder1dOption(
            in_channels=7,
            conv_params=interleave_arrays(
                [{"kernel_size": [3], "stride": [1, 2], "padding": [1]}] * 3,
                [{"kernel_size": [3], "stride": [1], "padding": [1]}] * motion_encoder_num_layers,
            ),
        )
    elif motion_encoder_name == "guided1d":
        motion_encoder_option = MotionGuidedEncoder1dOption(
            in_channels=7,
            conv_params=interleave_arrays(
                [{"kernel_size": [3], "stride": [2], "padding": [1]}] * 3,
                [{"kernel_size": [3], "stride": [1], "padding": [1]}] * motion_encoder_num_layers,
            ),
        )
    elif motion_encoder_name == "normal1d":
        motion_encoder_option = MotionNormalEncoder1dOption(
            in_channels=7,
            conv_params=interleave_arrays(
                [{"kernel_size": [3], "stride": [2], "padding": [1]}] * 3,
                [{"kernel_size": [3], "stride": [1], "padding": [1]}] * motion_encoder_num_layers,
                ),
        )
    elif motion_encoder_name == "tsn1d":
        motion_encoder_option = MotionTSNEncoder1dOption(
            in_channels=7,
            conv_params=interleave_arrays(
                [{"kernel_size": [3], "stride": [2], "padding": [1]}] * 3,
                [{"kernel_size": [3], "stride": [1], "padding": [1]}] * motion_encoder_num_layers,
            ),
        )
    else:
        raise RuntimeError("unreachable")
    latent_dim = int(trial.suggest_discrete_uniform("latent_dim", 16, 64, 8))
    content_encoder_num_layers = trial.suggest_int("content_encoder_encoder_num_layers", 0, 3)
    if network_name == "rae2d":
        network_option = RAE2dOption(
            latent_dim=latent_dim,
            conv_params=interleave_arrays(
                [{"kernel_size": [3], "stride": [2], "padding": [1]}] * 3,
                [{"kernel_size": [3], "stride": [1], "padding": [1]}] * content_encoder_num_layers,
            ),
            motion_encoder=motion_encoder_option,
            upsample_size=[8, 8],
        )
    elif network_name == "rdae2d":
        network_option = RDAE2dOption(
            latent_dim=latent_dim,
            conv_params=interleave_arrays(
                [{"kernel_size": [3], "stride": [2], "padding": [1]}] * 3,
                [{"kernel_size": [3], "stride": [1], "padding": [1]}] * content_encoder_num_layers,
            ),
            motion_encoder=motion_encoder_option,
            upsample_size=[8, 8],
            aggregation_method=trial.suggest_categorical("aggregation_method", ["concat", "sum"]),
        )
    elif network_name == "hrdae2d":
        network_option = HRDAE2dOption(
            latent_dim=latent_dim,
            conv_params=interleave_arrays(
                [{"kernel_size": [3], "stride": [2], "padding": [1]}] * 3,
                [{"kernel_size": [3], "stride": [1], "padding": [1]}] * content_encoder_num_layers,
                ),
            motion_encoder=motion_encoder_option,
            upsample_size=[8, 8],
            aggregation_method=trial.suggest_categorical("aggregation_method", ["concat", "sum"]),
        )
    else:
        raise RuntimeError("unreachable")

    optimizer_option = AdamOptimizerOption(
        lr=trial.suggest_loguniform("lr", 1e-5, 1e-2),
    )

    scheduler_option = OneCycleLRSchedulerOption(
        max_lr=trial.suggest_loguniform("max_lr", 1e-3, 1e-2),
    )

    model_option = VRModelOption(
        loss_coef={"wmse": 1.0},
        phase=phase,
        pred_diff=trial.suggest_categorical("pred_diff", [True, False]),
        loss=loss_option,
        network=network_option,
        optimizer=optimizer_option,
        scheduler=scheduler_option,
    )

    result_dir = Path(f"result/tuning/mmnist/{network_name}/{motion_encoder_name}/{trial.number}")
    train_option = TrainExpOption(
        result_dir=result_dir,
        dataloader=dataloader_option,
        model=model_option,
        n_epoch=50,
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "config.json", "w") as f:
        json.dump(asdict(train_option), f, indent=2)

    train_loader, val_loader = create_dataloader(
        dataloader_option,
        is_train=True,
    )
    model = create_model(
        model_option,
        n_epoch=train_option.n_epoch,
        steps_per_epoch=len(train_loader),
    )
    return model.train(
        train_loader,
        val_loader,
        n_epoch=train_option.n_epoch,
        result_dir=result_dir,
    )


if __name__ == "__main__":
    import optuna

    study = optuna.create_study(
        study_name="tuning",
        storage="sqlite:///result/tuning.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100)
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)
    study.trials_dataframe().to_csv("result/tuning.csv")
