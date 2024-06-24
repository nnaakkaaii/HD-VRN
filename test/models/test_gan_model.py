from tempfile import TemporaryDirectory
from pathlib import Path

from torch import rand, Tensor, nn
from torch.utils.data import Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from hrdae.models.gan_model import GANModel
from hrdae.models.losses import LossMixer, TemporalSimilarityLossOption, create_loss


class FakeDataset(Dataset):
    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "x": rand((10, 1, 16, 16)),
            "t": rand((10, 1, 16, 16)),
        }

    def __len__(self) -> int:
        return 10


class FakeGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 2, 1),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(4, 4, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 4, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, 3, 2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.conv(x)
        return self.deconv(latent), latent


class FakeDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x).mean(2).mean(2)


def test_basic_model():

    generator = FakeGenerator()
    discriminator = FakeDiscriminator()
    optimizer_g = Adam(generator.parameters())
    optimizer_d = Adam(discriminator.parameters())
    scheduler_g = StepLR(optimizer_g, step_size=1)
    scheduler_d = StepLR(optimizer_d, step_size=1)
    criterion_g = LossMixer(
        {
            "bce": nn.BCEWithLogitsLoss(),
            "tsim": create_loss(TemporalSimilarityLossOption()),
        },
        {"bce": 0.5, "tsim": 0.5},
    )
    criterion_d = LossMixer(
        {"bce": nn.BCEWithLogitsLoss()},
        {"bce": 1},
    )
    dataloader = DataLoader(FakeDataset(), batch_size=4)

    model = GANModel(
        generator,
        discriminator,
        optimizer_g,
        optimizer_d,
        scheduler_g,
        scheduler_d,
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
