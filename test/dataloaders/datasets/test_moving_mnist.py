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
        )
        data = dataset[0]
        assert data["x-"].shape == (10, 1, 64)
        assert data["x-_0"].shape == (1, 64)
        assert data["x-_t"].shape == (1, 64)
        assert data["x-_all"].shape == (2, 64)
        assert data["x+"].shape == (10, 1, 64, 64)
        assert data["x+_0"].shape == (1, 64, 64)
        assert data["x+_t"].shape == (1, 64, 64)
        assert data["x+_all"].shape == (2, 64, 64)
        assert data["slice_idx"].shape == (1,)
        assert data["idx_expanded"].shape == (10, 1, 64)
