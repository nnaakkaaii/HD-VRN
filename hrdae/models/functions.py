from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save_images(
    original: np.ndarray,
    reconstructed: np.ndarray,
    save_path: Path,
) -> None:
    # (b, 1, h, w)
    save_path.parent.mkdir(parents=True, exist_ok=True)
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
    plt.savefig(save_path)
    plt.close()


def save_reconstructed_images(
    original: np.ndarray, reconstructed: np.ndarray, epoch: int, save_dir: Path
):
    if len(original.shape) == 4:
        _save_images(
            original,
            reconstructed,
            save_dir / f"epoch_{epoch}.png",
        )
    elif len(original.shape) == 5:
        b, _, _, _, _ = original.shape
        for bi in range(0, b, 5):
            _save_images(
                original[bi],  # (t, 1, h, w)
                reconstructed[bi],  # (t, 1, h, w)
                save_dir / f"epoch_{epoch}_batch_{bi}.png",
            )
    elif len(original.shape) == 6:
        b, _, _, d, h, w = original.shape
        for bi in range(0, b, 5):
            _save_images(
                original[bi, :, :, :, :, w // 2],  # (t, 1, d, h)
                reconstructed[bi, :, :, :, :, w // 2],  # (t, 1, d, h)
                save_dir / f"epoch_{epoch}_batch_{bi}_axis_x.png",
            )
            _save_images(
                original[bi, :, :, :, h // 2],  # (t, 1, d, w)
                reconstructed[bi, :, :, :, h // 2],  # (t, 1, d, w)
                save_dir / f"epoch_{epoch}_batch_{bi}_axis_y.png",
            )
            _save_images(
                original[bi, :,  :, d // 2],  # (t, 1, h, w)
                reconstructed[bi, :, :, d // 2],  # (t, 1, h, w)
                save_dir / f"epoch_{epoch}_batch_{bi}_axis_z.png",
            )
    else:
        raise ValueError("Invalid shape of original and reconstructed videos")
