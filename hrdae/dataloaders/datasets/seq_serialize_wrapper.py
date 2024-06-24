from torch import Tensor
from torch.utils.data import Dataset


class SeqSerializeWrapper(Dataset):
    def __init__(
        self,
        base: Dataset,
    ) -> None:
        super().__init__()

        self.base = base

    def __len__(self) -> int:
        return len(self.base)  # type: ignore

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        x = self.base[index]["xp"]
        b = x.size(0)
        n = x.size(1)
        size = x.size()[2:]
        return {
            "x": x.reshape(b * n, *size),
            "t": x.reshape(b * n, *size),
        }
