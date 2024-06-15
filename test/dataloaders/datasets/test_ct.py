from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from torchvision import transforms

from hrdae.dataloaders.datasets.ct import CT, BasicSliceIndexer
from hrdae.dataloaders.transforms import (
    MinMaxNormalizationOption,
    Pool3dOption,
    UniformShape3dOption,
    RandomShift3dOption,
    create_transform,
)


def test_CT():
    with TemporaryDirectory() as root:
        data_root = Path(root) / "CT"
        data_root.mkdir(parents=True, exist_ok=True)
        np.savez(
            data_root / "sample1.npz",
            np.random.randn(10, 10, 16, 16),
        )
        dataset = CT(
            root=Path(root),
            slice_indexer=BasicSliceIndexer(),
            transform=transforms.Compose(
                [
                    create_transform(MinMaxNormalizationOption()),
                    create_transform(
                        UniformShape3dOption(
                            target_shape=[16, 32, 32],
                        )
                    ),
                    create_transform(
                        RandomShift3dOption(
                            max_shifts=[2, 2, 2],
                        )
                    ),
                    create_transform(
                        Pool3dOption(
                            pool_size=[1, 2, 2],
                        )
                    ),
                ]
            ),
            in_memory=True,
            is_train=False,
        )
        data = dataset[0]
        assert data["x-"].shape == (10, 1, 16, 16)
        assert data["x-_0"].shape == (1, 16, 16)
        assert data["x-_t"].shape == (1, 16, 16)
        assert data["x-_all"].shape == (2, 16, 16)
        assert data["x+"].shape == (10, 1, 16, 16, 16)
        assert data["x+_0"].shape == (1, 16, 16, 16)
        assert data["x+_t"].shape == (1, 16, 16, 16)
        assert data["x+_all"].shape == (2, 16, 16, 16)
        assert data["slice_idx"].shape == (1,)
        assert data["idx_expanded"].shape == (10, 1, 16, 16)
