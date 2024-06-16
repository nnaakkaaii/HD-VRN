import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from hrdae.option import TrainExpOption
from hrdae.dataloaders.transforms import (
    RandomShift3dOption,
    Pool3dOption,
)
from hrdae.dataloaders.datasets import CTDatasetOption
from hrdae.dataloaders import create_dataloader, BasicDataLoaderOption
from hrdae.models import create_model, VRModelOption
from hrdae.models.losses import WeightedMSELossOption
from hrdae.models.optimizers import AdamOptimizerOption
from hrdae.models.schedulers import OneCycleLRSchedulerOption
from hrdae.models.networks import RAE3dOption, RDAE3dOption, HRDAE3dOption
from hrdae.models.networks.rnn import ConvLSTM2dOption, GRU2dOption, TCN2dOption
from hrdae.models.networks.motion_encoder import (
    MotionRNNEncoder2dOption,
    MotionConv3dEncoder2dOption,
    MotionGuidedEncoder2dOption,
    MotionNormalEncoder2dOption,
    MotionTSNEncoder2dOption,
)


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
            result.extend(am[am_index : am_index + num_elements])
            am_index += num_elements
            if extra > 0:
                extra -= 1
    return result


def default(item: Any):
    match item:
        case Path():
            return str(item)
        case _:
            raise TypeError(type(item))


def objective(trial):
    dataset_option = CTDatasetOption(
        root=Path("data"),
        slice_index=[48, 80],
        threshold=0.1,
        min_occupancy=0.2,
        in_memory=False,
    )

    pool_size = [2, 2, 2]
    d, h, w = 64 // pool_size[0], 128 // pool_size[1], 128 // pool_size[2]
    transform_option = {
        "random_shift3d": RandomShift3dOption(
            max_shifts=[2, 4, 4],
        ),
        "pool3d": Pool3dOption(
            pool_size=pool_size,
        ),
    }
    num_reducible_layers = trial.suggest_int("num_reducible_layers", 1, 3)
    d_, h_, w_ = d // 2 ** num_reducible_layers, h // 2 ** num_reducible_layers, w // 2 ** num_reducible_layers

    dataloader_option = BasicDataLoaderOption(
        batch_size=16,
        train_val_ratio=0.8,
        dataset=dataset_option,
        transform_order_train=["random_shift3d", "pool3d"],
        transform_order_val=["pool3d"],
        transform=transform_option,
    )

    loss_option = {
        "wmse": WeightedMSELossOption(
            weight_dynamic=1.0,
        ),
    }

    phase = trial.suggest_categorical("phase", ["0", "t", "all"])
    if args.network_name in [
        "hrdae3d",
        "rae3d",
        "rdae3d",
    ]:
        network_name = args.network_name
    else:
        network_name = trial.suggest_categorical(
            "network", ["hrdae3d", "rae3d", "rdae3d"]
        )
    if args.motion_encoder_name in [
        "conv3d",
        "guided2d",
        "normal2d",
        "rnn2d",
        "tsn2d",
    ]:
        if phase == "all":
            pred_diff = False
        else:
            pred_diff = trial.suggest_categorical("pred_diff", [True, False])
        motion_encoder_name = args.motion_encoder_name
    elif phase == "all":
        pred_diff = False
        motion_encoder_name = trial.suggest_categorical(
            "motion_encoder_all", ["conv3d", "normal2d", "rnn2d"]
        )
    else:
        pred_diff = trial.suggest_categorical("pred_diff", [True, False])
        motion_encoder_name = trial.suggest_categorical(
            "motion_encoder", ["conv3d", "guided2d", "normal2d", "rnn2d", "tsn2d"]
        )
    motion_encoder_num_layers = trial.suggest_int("motion_encoder_num_layers", 0, 6)
    if motion_encoder_name == "rnn2d":
        if args.rnn_name in [
            "conv_lstm2d",
            "gru2d",
            "tcn2d",
        ]:
            rnn_name = args.rnn_name
        else:
            rnn_name = trial.suggest_categorical(
                "rnn", ["conv_lstm2d", "gru2d", "tcn2d"]
            )
        motion_encoder_name = f"{motion_encoder_name}/{rnn_name}"
        rnn_num_layers = trial.suggest_int("rnn_num_layers", 1, 4)
        if rnn_name == "conv_lstm2d":
            rnn_option = ConvLSTM2dOption(
                num_layers=rnn_num_layers,
            )
        elif rnn_name == "gru2d":
            rnn_option = GRU2dOption(
                num_layers=rnn_num_layers,
                image_size=[d_, h_],
            )
        elif rnn_name == "tcn2d":
            rnn_option = TCN2dOption(
                num_layers=rnn_num_layers,
                image_size=[d_, h_],
                kernel_size=4,
                dropout=0.1,
            )
        else:
            raise RuntimeError("unreachable")
        motion_encoder_option = MotionRNNEncoder2dOption(
            in_channels=2,
            conv_params=interleave_arrays(
                [{"kernel_size": [3], "stride": [2], "padding": [1]}]
                * num_reducible_layers,
                [{"kernel_size": [3], "stride": [1], "padding": [1]}]
                * motion_encoder_num_layers,
            ),
            rnn=rnn_option,
        )
    elif motion_encoder_name == "conv3d":
        motion_encoder_option = MotionConv3dEncoder2dOption(
            in_channels=2,
            conv_params=interleave_arrays(
                [{"kernel_size": [3], "stride": [1, 2, 2], "padding": [1]}]
                * num_reducible_layers,
                [{"kernel_size": [3], "stride": [1], "padding": [1]}]
                * motion_encoder_num_layers,
            ),
        )
    elif motion_encoder_name == "guided2d":
        motion_encoder_option = MotionGuidedEncoder2dOption(
            in_channels=2,
            conv_params=interleave_arrays(
                [{"kernel_size": [3], "stride": [2], "padding": [1]}]
                * num_reducible_layers,
                [{"kernel_size": [3], "stride": [1], "padding": [1]}]
                * motion_encoder_num_layers,
            ),
        )
    elif motion_encoder_name == "normal2d":
        motion_encoder_option = MotionNormalEncoder2dOption(
            in_channels=2,
            conv_params=interleave_arrays(
                [{"kernel_size": [3], "stride": [2], "padding": [1]}]
                * num_reducible_layers,
                [{"kernel_size": [3], "stride": [1], "padding": [1]}]
                * motion_encoder_num_layers,
            ),
        )
    elif motion_encoder_name == "tsn2d":
        motion_encoder_option = MotionTSNEncoder2dOption(
            in_channels=2,
            conv_params=interleave_arrays(
                [{"kernel_size": [3], "stride": [2], "padding": [1]}]
                * num_reducible_layers,
                [{"kernel_size": [3], "stride": [1], "padding": [1]}]
                * motion_encoder_num_layers,
            ),
        )
    else:
        raise RuntimeError("unreachable")
    latent_dim = trial.suggest_int("latent_dim", 16, 256, step=16)
    content_encoder_num_layers = trial.suggest_int(
        "content_encoder_encoder_num_layers", 0, 5
    )
    if network_name == "rae3d":
        network_option = RAE3dOption(
            latent_dim=latent_dim,
            conv_params=interleave_arrays(
                [{"kernel_size": [3], "stride": [2], "padding": [1]}]
                * num_reducible_layers,
                [{"kernel_size": [3], "stride": [1], "padding": [1]}]
                * content_encoder_num_layers,
            ),
            motion_encoder=motion_encoder_option,
            upsample_size=[d_, h_, w_],
        )
    elif network_name == "rdae3d":
        network_option = RDAE3dOption(
            latent_dim=latent_dim,
            conv_params=interleave_arrays(
                [{"kernel_size": [3], "stride": [2], "padding": [1]}]
                * num_reducible_layers,
                [{"kernel_size": [3], "stride": [1], "padding": [1]}]
                * content_encoder_num_layers,
            ),
            motion_encoder=motion_encoder_option,
            upsample_size=[d_, h_, w_],
            aggregation_method=trial.suggest_categorical(
                "aggregation_method", ["concat", "sum"]
            ),
        )
    elif network_name == "hrdae3d":
        network_option = HRDAE3dOption(
            latent_dim=latent_dim,
            conv_params=interleave_arrays(
                [{"kernel_size": [3], "stride": [2], "padding": [1]}]
                * num_reducible_layers,
                [{"kernel_size": [3], "stride": [1], "padding": [1]}]
                * content_encoder_num_layers,
            ),
            motion_encoder=motion_encoder_option,
            upsample_size=[d_, h_, w_],
            aggregation_method=trial.suggest_categorical(
                "aggregation_method", ["concat", "sum"]
            ),
        )
    else:
        raise RuntimeError("unreachable")

    optimizer_option = AdamOptimizerOption(
        lr=trial.suggest_float("lr", 1e-5, 1e-2, log=True),
    )

    scheduler_option = OneCycleLRSchedulerOption(
        max_lr=trial.suggest_float("max_lr", 1e-3, 1e-2, log=True),
    )

    model_option = VRModelOption(
        loss_coef={"wmse": 1.0},
        phase=phase,
        pred_diff=pred_diff,
        loss=loss_option,
        network=network_option,
        optimizer=optimizer_option,
        scheduler=scheduler_option,
    )

    result_dir = Path(
        f"results/tuning/ct/{network_name}/{motion_encoder_name}/{trial.number}-{str(uuid4())[:8]}"
    )
    train_option = TrainExpOption(
        result_dir=result_dir,
        dataloader=dataloader_option,
        model=model_option,
        n_epoch=100,
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / "config.json", "w") as f:
        json.dump(asdict(train_option), f, indent=2, default=default)

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
        debug=False,
    )


if __name__ == "__main__":
    import argparse

    import optuna

    parser = argparse.ArgumentParser()
    parser.add_argument("--network_name", type=str)
    parser.add_argument("--motion_encoder_name", type=str)
    parser.add_argument("--rnn_name", type=str)
    args = parser.parse_args()

    study_name = "ct"
    if args.network_name is not None:
        assert args.network_name in [
            "hrdae3d",
            "rae3d",
            "rdae3d",
        ]
        study_name += f"_{args.network_name}"
    if args.motion_encoder_name is not None:
        assert args.motion_encoder_name in [
            "conv3d",
            "guided2d",
            "normal2d",
            "rnn2d",
            "tsn2d",
        ]
        study_name += f"_{args.motion_encoder_name}"
    if args.rnn_name is not None:
        assert args.rnn_name in [
            "conv_lstm2d",
            "gru2d",
            "tcn2d",
        ]
        study_name += f"_{args.rnn_name}"
    study = optuna.create_study(
        study_name=study_name,
        storage="sqlite:///results/tuning/ct/sqlite.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=500)
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)
    study.trials_dataframe().to_csv("results/tuning/ct/trials.csv")
