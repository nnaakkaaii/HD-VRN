from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn


def _save_images(
    original: np.ndarray,
    reconstructed: np.ndarray,
    save_dir: Path,
    basename: str,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    # single image
    for i in range(10):
        if i >= len(original):
            break
        for image, name in zip(
            [original, reconstructed],
            ["original", "reconstructed"],
        ):
            save_path = save_dir / f"{basename}_{name}_{i}.png"

            plt.figure(figsize=(5, 5))
            plt.imshow(np.squeeze(image[i]), cmap="gray")
            plt.axis("off")
            plt.savefig(save_path)
            plt.close()

    # concatenated image
    # (b, 1, h, w)
    plt.figure(figsize=(20, 4))

    # Original Images
    for i in range(10):
        if i >= len(original):
            break
        plt.subplot(2, 10, i + 1)
        plt.imshow(np.squeeze(original[i]), cmap="gray")
        plt.title("Original")
        plt.axis("off")

    # Reconstructed Images
    for i in range(10):
        if i >= len(reconstructed):
            break
        plt.subplot(2, 10, i + 11)
        plt.imshow(np.squeeze(reconstructed[i]), cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")

    # 保存するファイル名を設定
    plt.savefig(save_dir / f"{basename}.png")
    plt.close()


def save_reconstructed_images(
    original: np.ndarray, reconstructed: np.ndarray, name: str, save_dir: Path
):
    if len(original.shape) == 4:
        _save_images(
            original,
            reconstructed,
            save_dir,
            name,
        )
    elif len(original.shape) == 5:
        b, _, _, _, _ = original.shape
        for bi in range(b):
            _save_images(
                original[bi],  # (t, 1, h, w)
                reconstructed[bi],  # (t, 1, h, w)
                save_dir,
                f"{name}_batch_{bi}",
            )
    elif len(original.shape) == 6:
        b, _, _, d, h, w = original.shape
        for bi in range(b):
            _save_images(
                original[bi, :, :, :, :, w // 2],  # (t, 1, d, h)
                reconstructed[bi, :, :, :, :, w // 2],  # (t, 1, d, h)
                save_dir,
                f"{name}_batch_{bi}_axis_x",
            )
            _save_images(
                original[bi, :, :, :, h // 2],  # (t, 1, d, w)
                reconstructed[bi, :, :, :, h // 2],  # (t, 1, d, w)
                save_dir,
                f"{name}_batch_{bi}_axis_y",
            )
            _save_images(
                original[bi, :, :, d // 2],  # (t, 1, h, w)
                reconstructed[bi, :, :, d // 2],  # (t, 1, h, w)
                save_dir,
                f"{name}_batch_{bi}_axis_z",
            )
    else:
        raise ValueError("Invalid shape of original and reconstructed videos")


def save_model(model: nn.Module, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(model_to_save.state_dict(), filepath)


def shuffled_indices(length: int) -> Tensor:
    indices = torch.empty(length, dtype=torch.long)
    for i in range(length):
        choices = torch.cat([torch.arange(0, i), torch.arange(i + 1, length)])
        indices[i] = choices[torch.randint(len(choices), (1,))]
    return indices
