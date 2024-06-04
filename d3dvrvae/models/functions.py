from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def save_reconstructed_images(
        original: np.ndarray, reconstructed: np.ndarray, epoch: int, save_dir: Path
):
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(20, 4))

    # Original Images
    for i in range(10):
        plt.subplot(2, 10, i + 1)
        plt.imshow(original[i].reshape(28, 28), cmap="gray")
        plt.title("Original")
        plt.axis("off")

    # Reconstructed Images
    for i in range(10):
        plt.subplot(2, 10, i + 11)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")

    # 保存するファイル名を設定
    plt.savefig(save_dir / f"original_epoch_{epoch}.png")
    plt.close()
