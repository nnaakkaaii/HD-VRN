from tempfile import TemporaryDirectory
from pathlib import Path

from torch import rand, Tensor, nn
from torch.utils.data import Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from hrdae.models.basic_model import BasicModel
from hrdae.models.losses import LossMixer


class FakeDataset(Dataset):
    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return {
            "xp": rand((1, 32, 32)),
        }

    def __len__(self) -> int:
        return 10


class FakeNetwork(nn.Module):
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


def test_basic_model():

    network = FakeNetwork()
    optimizer = Adam(network.parameters())
    scheduler = StepLR(optimizer, step_size=1)
    criterion = LossMixer(
        {"mse": nn.MSELoss()},
        {"mse": 1},
    )
    dataloader = DataLoader(FakeDataset(), batch_size=4)

    model = BasicModel(network, optimizer, scheduler, criterion)
    with TemporaryDirectory() as tempdir:
        model.train(
            dataloader,
            dataloader,
            1,
            Path(tempdir),
            False,
        )
