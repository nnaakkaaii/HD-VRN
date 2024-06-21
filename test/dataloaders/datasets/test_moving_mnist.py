from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from hrdae.dataloaders.datasets.moving_mnist import MovingMNIST


def test_MovingMNIST():
    with TemporaryDirectory() as root:
        data_root = Path(root) / "MovingMNIST"
        data_root.mkdir(parents=True, exist_ok=True)
        np.save(
            str(data_root / "mnist_test_seq.npy"),
            np.random.randn(20, 10, 64, 64),
        )
        dataset = MovingMNIST(
            root=root,
            slice_index=[32],
            split="test",
            download=False,
            transform=None,
            content_phase="all",
            motion_phase="all",
            motion_aggregator="concat",
        )
        data = dataset[0]
        assert data["xm"].shape == (10, 3, 64)  # 3: phase=all, agg=concat
        assert data["xm_0"].shape == (2, 64)
        assert data["xp"].shape == (10, 1, 64, 64)
        assert data["xp_0"].shape == (2, 64, 64)
        assert data["slice_idx"].shape == (1,)
        assert data["idx_expanded"].shape == (10, 1, 64)
