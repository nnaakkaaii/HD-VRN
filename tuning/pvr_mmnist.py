import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from hrdae.option import TrainExpOption
from hrdae.dataloaders.transforms import MinMaxNormalizationOption
from hrdae.dataloaders.datasets import MovingMNISTDatasetOption
from hrdae.dataloaders import create_dataloader, BasicDataLoaderOption
from hrdae.models import create_model, PVRModelOption
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
    motion_aggregation = trial.suggest_categorical(
        "motion_aggregation", ["concat", "diff"]
    )
    if motion_aggregation == "concat":
        motion_phase = trial.suggest_categorical(
            "motion_phase__concat", ["none", "0", "t", "all"]
        )
    elif motion_aggregation == "diff":
        motion_phase = trial.suggest_categorical("motion_phase__diff", ["0", "t"])
    else:
        raise RuntimeError("unreachable")
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
    motion_encoder_num_layers = trial.suggest_int("motion_encoder_num_layers", 0, 6)
    if args.motion_encoder_name in [
        "conv2d",
        "guided1d",
        "normal1d",
        "rnn1d",
    ]:
        motion_encoder_name = args.motion_encoder_name
    else:
        motion_encoder_name = trial.suggest_categorical(
            "motion_encoder", ["conv2d", "guided1d", "normal1d", "rnn1d"]
        )
    if motion_encoder_name == "rnn1d":
        rnn_num_layers = trial.suggest_int("rnn_num_layers", 1, 5)
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
    else:
        rnn_num_layers = 0
        rnn_name = ""

    content_encoder_grad = trial.suggest_categorical(
        "content_encoder_grad", [True, False]
    )
    decoder_grad = trial.suggest_categorical("decoder_grad", [True, False])
    if content_encoder_grad:
        content_encoder_lr = trial.suggest_float(
            "content_encoder_lr", 1e-6, 1e-2, log=True
        )
    else:
        content_encoder_lr = 0
    if decoder_grad:
        decoder_lr = trial.suggest_float("decoder_lr", 1e-6, 1e-2, log=True)
    else:
        decoder_lr = 0

    dataset_option = MovingMNISTDatasetOption(
        root="data",
        slice_index=[16, 32, 48],
        content_phase="0",
        motion_phase=motion_phase,
        motion_aggregation=motion_aggregation,
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

    if motion_encoder_name == "rnn1d":
        motion_encoder_name = f"{motion_encoder_name}/{rnn_name}"
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

    latent_dim = 8
    content_encoder_num_layers = 0
    aggregation_method = "sum"

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
            aggregation_method=aggregation_method,
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
            aggregation_method=aggregation_method,
        )
    else:
        raise RuntimeError("unreachable")

    lr = 0.005
    max_lr = 0.05
    optimizer_option = AdamOptimizerOption(
        lrs={
            "content_encoder": content_encoder_lr,
            "decoder": decoder_lr,
            "motion_encoder": lr,
        },
    )

    scheduler_option = OneCycleLRSchedulerOption(
        max_lr=max_lr,
    )

    model_option = PVRModelOption(
        network_grad={
            "content_encoder": content_encoder_grad,
            "decoder": decoder_grad,
            "motion_encoder": True,
        },
        network_weight={
            "content_encoder": args.encoder_weight,
            "decoder": args.decoder_weight,
        },
        loss_coef={"wmse": 1.0},
        loss=loss_option,
        network=network_option,
        optimizer=optimizer_option,
        scheduler=scheduler_option,
    )

    result_dir = Path(
        f"results/tuning/pvr/mmnist/{network_name}/{motion_encoder_name}/{trial.number}-{str(uuid4())[:8]}"
    )
    train_option = TrainExpOption(
        result_dir=result_dir,
        dataloader=dataloader_option,
        model=model_option,
        n_epoch=10,
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
    parser.add_argument("--encoder_weight", type=str, required=True)
    parser.add_argument("--decoder_weight", type=str, required=True)
    parser.add_argument("--network_name", type=str)
    parser.add_argument("--motion_encoder_name", type=str)
    parser.add_argument("--rnn_name", type=str)
    parser.add_argument("--weight", type=float, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
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
        storage="sqlite:///results/tuning/mmnist/pvr/sqlite.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100)
    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)
    study.trials_dataframe().to_csv("results/tuning/mmnist/pvr/trials.csv")
