import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

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
from hrdae.models.networks.motion_encoder import (
    MotionRNNEncoder1dOption,
    MotionConv2dEncoder1dOption,
    MotionGuidedEncoder1dOption,
    MotionNormalEncoder1dOption,
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
    dataset_option = MovingMNISTDatasetOption(
        root="data",
        slice_index=[16, 32, 48],
    )

    transform_option = {
        "min_max_normalization": MinMaxNormalizationOption(),
    }
    num_reducible_layers = 4

    dataloader_option = BasicDataLoaderOption(
        batch_size=args.batch_size,
        train_val_ratio=0.8,
        dataset=dataset_option,
        transform_order_train=["min_max_normalization"],
        transform_order_val=["min_max_normalization"],
        transform=transform_option,
    )

    loss_option = {
        "wmse": WeightedMSELossOption(
            weight_dynamic=args.weight,
        ),
    }

    # phase = trial.suggest_categorical("phase", ["0", "all"])
    phase = "all"
    if args.network_name in [
        "hrdae2d",
        "rae2d",
        "rdae2d",
    ]:
        network_name = args.network_name
    else:
        network_name = trial.suggest_categorical(
            "network", ["hrdae2d", "rae2d", "rdae2d"]
        )
    if args.motion_encoder_name in [
        "conv2d",
        "guided1d",
        "normal1d",
        "rnn1d",
        "tsn1d",
    ]:
        # if phase == "all":
        #     pred_diff = False
        # else:
        #     pred_diff = trial.suggest_categorical("pred_diff", [True, False])
        motion_encoder_name = args.motion_encoder_name
    elif phase == "all":
        # pred_diff = False
        motion_encoder_name = trial.suggest_categorical(
            "motion_encoder_all", ["conv2d", "normal1d", "rnn1d"]
        )
    else:
        # pred_diff = trial.suggest_categorical("pred_diff", [True, False])
        motion_encoder_name = trial.suggest_categorical(
            "motion_encoder", ["conv2d", "guided1d", "normal1d", "rnn1d", "tsn1d"]
        )
    motion_encoder_num_layers = 0
    # motion_encoder_num_layers = trial.suggest_int("motion_encoder_num_layers", 0, 6)
    if motion_encoder_name == "rnn1d":
        if args.rnn_name in [
            "conv_lstm1d",
            "gru1d",
            "tcn1d",
        ]:
            rnn_name = args.rnn_name
        else:
            rnn_name = trial.suggest_categorical(
                "rnn", ["conv_lstm1d", "gru1d", "tcn1d"]
            )
        motion_encoder_name = f"{motion_encoder_name}/{rnn_name}"
        rnn_num_layers = 3
        # rnn_num_layers = trial.suggest_int("rnn_num_layers", 1, 5)
        if rnn_name == "conv_lstm1d":
            rnn_option = ConvLSTM1dOption(
                num_layers=rnn_num_layers,
            )
        elif rnn_name == "gru1d":
            rnn_option = GRU1dOption(
                num_layers=rnn_num_layers,
                image_size=64 // 2**num_reducible_layers,
            )
        elif rnn_name == "tcn1d":
            rnn_option = TCN1dOption(
                num_layers=rnn_num_layers,
                image_size=64 // 2**num_reducible_layers,
                kernel_size=3,
                dropout=0.1,
            )
        else:
            raise RuntimeError("unreachable")
        motion_encoder_option = MotionRNNEncoder1dOption(
            in_channels=3,
            conv_params=interleave_arrays(
                [{"kernel_size": [3], "stride": [2], "padding": [1]}]
                * num_reducible_layers,
                [{"kernel_size": [3], "stride": [1], "padding": [1]}]
                * motion_encoder_num_layers,
            ),
            rnn=rnn_option,
        )
    elif motion_encoder_name == "conv2d":
        motion_encoder_option = MotionConv2dEncoder1dOption(
            in_channels=3,
            conv_params=interleave_arrays(
                [{"kernel_size": [3], "stride": [1, 2], "padding": [1]}]
                * num_reducible_layers,
                [{"kernel_size": [3], "stride": [1], "padding": [1]}]
                * motion_encoder_num_layers,
            ),
        )
    elif motion_encoder_name == "guided1d":
        motion_encoder_option = MotionGuidedEncoder1dOption(
            in_channels=3,
            conv_params=interleave_arrays(
                [{"kernel_size": [3], "stride": [2], "padding": [1]}]
                * num_reducible_layers,
                [{"kernel_size": [3], "stride": [1], "padding": [1]}]
                * motion_encoder_num_layers,
            ),
        )
    elif motion_encoder_name == "normal1d":
        motion_encoder_option = MotionNormalEncoder1dOption(
            in_channels=3,
            conv_params=interleave_arrays(
                [{"kernel_size": [3], "stride": [2], "padding": [1]}]
                * num_reducible_layers,
                [{"kernel_size": [3], "stride": [1], "padding": [1]}]
                * motion_encoder_num_layers,
            ),
        )
    else:
        raise RuntimeError("unreachable")
    # latent_dim = 128
    latent_dim = trial.suggest_int("latent_dim", 32, 256, step=32)
    content_encoder_num_layers = 0
    # content_encoder_num_layers = trial.suggest_int(
    #     "content_encoder_num_layers", 0, 4
    # )
    if network_name == "rae2d":
        network_option = RAE2dOption(
            latent_dim=latent_dim,
            conv_params=interleave_arrays(
                [{"kernel_size": [3], "stride": [2], "padding": [1]}]
                * num_reducible_layers,
                [{"kernel_size": [3], "stride": [1], "padding": [1]}]
                * content_encoder_num_layers,
            ),
            motion_encoder=motion_encoder_option,
            upsample_size=[
                64 // 2**num_reducible_layers,
                64 // 2**num_reducible_layers,
            ],
        )
    elif network_name == "rdae2d":
        network_option = RDAE2dOption(
            latent_dim=latent_dim,
            conv_params=interleave_arrays(
                [{"kernel_size": [3], "stride": [2], "padding": [1]}]
                * num_reducible_layers,
                [{"kernel_size": [3], "stride": [1], "padding": [1]}]
                * content_encoder_num_layers,
            ),
            motion_encoder=motion_encoder_option,
            upsample_size=[
                64 // 2**num_reducible_layers,
                64 // 2**num_reducible_layers,
            ],
            # aggregation_method=trial.suggest_categorical(
            #     "aggregation_method", ["concat", "sum"]
            # ),
            aggregation_method="concat",
        )
    elif network_name == "hrdae2d":
        network_option = HRDAE2dOption(
            latent_dim=latent_dim,
            conv_params=interleave_arrays(
                [{"kernel_size": [3], "stride": [2], "padding": [1]}]
                * num_reducible_layers,
                [{"kernel_size": [3], "stride": [1], "padding": [1]}]
                * content_encoder_num_layers,
            ),
            motion_encoder=motion_encoder_option,
            upsample_size=[
                64 // 2**num_reducible_layers,
                64 // 2**num_reducible_layers,
            ],
            # aggregation_method=trial.suggest_categorical(
            #     "aggregation_method", ["concat", "sum"]
            # ),
            aggregation_method="concat",
        )
    else:
        raise RuntimeError("unreachable")

    optimizer_option = AdamOptimizerOption(
        lr=trial.suggest_float("lr", 1e-5, 5e-2, log=True),
    )

    scheduler_option = OneCycleLRSchedulerOption(
        # max_lr=trial.suggest_float("max_lr", 1e-3, 5e-2, log=True),
        max_lr=0.05
    )

    model_option = VRModelOption(
        loss_coef={"wmse": 1.0},
        phase=phase,
        pred_diff=False,
        loss=loss_option,
        network=network_option,
        optimizer=optimizer_option,
        scheduler=scheduler_option,
    )

    result_dir = Path(
        f"results/tuning/mmnist/{network_name}/{motion_encoder_name}/{trial.number}-{str(uuid4())[:8]}"
    )
    train_option = TrainExpOption(
        result_dir=result_dir,
        dataloader=dataloader_option,
        model=model_option,
        n_epoch=20,
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
    parser.add_argument("--network_name", type=str, default="hrdae2d")
    parser.add_argument("--motion_encoder_name", type=str, default="rnn1d")
    parser.add_argument("--rnn_name", type=str, default="tcn1d")
    parser.add_argument("--weight", type=float, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    study_name = "mmnist"
    if args.network_name is not None:
        assert args.network_name in [
            "hrdae2d",
            "rae2d",
            "rdae2d",
        ]
        study_name += f"_{args.network_name}"
    if args.motion_encoder_name is not None:
        assert args.motion_encoder_name in [
            "conv2d",
            "guided1d",
            "normal1d",
            "rnn1d",
            "tsn1d",
        ]
        study_name += f"_{args.motion_encoder_name}"
    if args.rnn_name is not None:
        assert args.rnn_name in [
            "conv_lstm1d",
            "gru1d",
            "tcn1d",
        ]
        study_name += f"_{args.rnn_name}"
    study_name += f"_w{args.weight}_dc1591a"
    study = optuna.create_study(
        study_name=study_name,
        storage="sqlite:///results/tuning/mmnist/sqlite.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100)
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)
    study.trials_dataframe().to_csv("results/tuning/mmnist/trials.csv")
