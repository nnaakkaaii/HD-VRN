from pathlib import Path
from tempfile import TemporaryDirectory

from torch import Tensor, nn, rand
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset

from hrdae.models.gan_model import GANModel
from hrdae.models.losses import (
    BCEWithLogitsLossOption,
    ContrastiveLossOption,
    LossMixer,
    MSELossOption,
    create_loss,
)


class FakeDataset(Dataset):
    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "xm": rand((10, 3, 32)),
            "xm_0": rand((2, 32)),
            "xp": rand((10, 1, 32, 32)),
            "xp_0": rand((2, 32, 32)),
        }

    def __len__(self) -> int:
        return 10


class FakeGenerator(nn.Module):
    def __init__(
        self,
        num_slices: int,
        content_phase: str = "all",
        motion_phase: str = "0",
        motion_aggregator: str = "concat",
    ) -> None:
        super().__init__()

        m = num_slices
        if motion_aggregator == "concat":
            if motion_phase in ["0", "t"]:
                m *= 2
            elif motion_phase == "all":
                m *= 3
        self.conv1d = nn.Sequential(
            nn.Conv1d(m, 4, 3, 2, 1),
            nn.ReLU(),
            nn.Conv1d(4, 4, 3, 2, 1),
            nn.ReLU(),
            nn.Conv1d(4, 4, 3, 2, 1),
        )
        c = 2 if content_phase == "all" else 1
        self.conv2d = nn.Sequential(
            nn.Conv2d(c, 4, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 2, 1),
        )
        self.deconv2d = nn.Sequential(
            nn.ConvTranspose2d(4, 4, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 4, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, 3, 2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        xm: Tensor,
        xp_0: Tensor,
        xm_0: Tensor,
    ) -> tuple[Tensor, list[Tensor], Tensor, list[Tensor]]:
        c = 4
        b, n, s, h = xm.size()
        _, _, _, w = xp_0.size()
        # xm (40, 1, 32)
        xm = xm.reshape(b * n, s, h)
        # ym (40, 4, 4)
        ym = self.conv1d(xm)
        # ym (4, 10, 4, 4, 1)
        ym = ym.reshape(b, n, c, h // 8, 1)

        # yp_0 (4, 4, 4, 4)
        yp_0 = self.conv2d(xp_0)
        # yp_0 (4, 1, 4, 4, 4)
        yp_0 = yp_0.reshape(b, 1, c, h // 8, w // 8)

        # y (4, 10, 4, 4, 4)
        y = ym + yp_0
        # y (40, 4, 4, 4)
        y_reshaped = y.reshape(b * n, c, h // 8, w // 8)

        # x (40, 1, 32, 32)
        x = self.deconv2d(y_reshaped)
        return x.reshape(b, n, 1, h, w), [ym], y, []


class FakeDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(8, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x).mean(2).mean(2)


def test_basic_model():

    generator = FakeGenerator(1, "all", "all", "concat")
    discriminator = FakeDiscriminator()
    optimizer_g = Adam(generator.parameters())
    optimizer_d = Adam(discriminator.parameters())
    scheduler_g = StepLR(optimizer_g, step_size=1)
    scheduler_d = StepLR(optimizer_d, step_size=1)
    criterion = LossMixer(
        {
            "mse": create_loss(MSELossOption()),
            "contrastive": create_loss(ContrastiveLossOption()),
        },
        {
            "mse": 0.5,
            "contrastive": 0.5,
        },
    )
    criterion_g = create_loss(BCEWithLogitsLossOption())
    criterion_d = create_loss(BCEWithLogitsLossOption())
    dataloader = DataLoader(FakeDataset(), batch_size=4)

    model = GANModel(
        generator,
        discriminator,
        optimizer_g,
        optimizer_d,
        scheduler_g,
        scheduler_d,
        criterion,
        criterion_g,
        criterion_d,
        serialize=True,
    )
    with TemporaryDirectory() as tempdir:
        model.train(
            dataloader,
            dataloader,
            1,
            Path(tempdir),
            False,
        )
