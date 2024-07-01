from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from torchvision import transforms

from hrdae.dataloaders.datasets.sliced_ct import SlicedCTDatasetOption, create_sliced_ct_dataset
from hrdae.dataloaders.transforms import (
    MinMaxNormalizationOption,
    Pool3dOption,
    UniformShape3dOption,
    RandomShift3dOption,
    create_transform,
)


def test_SlicedCT():
    with TemporaryDirectory() as root:
        data_root = Path(root) / "CT"
        data_root.mkdir(parents=True, exist_ok=True)
        np.savez(
            data_root / "sample1.npz",
            np.random.randn(10, 10, 16, 16),
        )
        print(data_root / "sample1.npz")
        opt = SlicedCTDatasetOption(
            root=Path(root),
            slice_index=[6, 10],
            in_memory=True,
            sequential=True,
            content_phase="0",
            motion_phase="0",
            motion_aggregation="none",
            slice_axis="z",
            slice_range=[2, 8],
        )
        transform = transforms.Compose(
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
        )
        dataset = create_sliced_ct_dataset(opt, transform, is_train=False)
        data = dataset[0]
        assert data["xm"].shape == (10, 2, 16)
        assert data["xm_0"].shape == (2, 16)
        assert data["xp"].shape == (10, 1, 16, 16)
        assert data["xp_0"].shape == (1, 16, 16)
