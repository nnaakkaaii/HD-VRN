from torch import Tensor
from torch.utils.data import Dataset


class SeqDivideWrapper(Dataset):
    def __init__(self,
                 base: Dataset,
                 period: int,
                 ) -> None:
        super().__init__()

        self.base = base
        self.period = period

    def __len__(self) -> int:
        return len(self.base) * self.period

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        x = self.base[index // self.period]['x+']
        return {
            "x": x[index % self.period],
            "t": x[index % self.period],
        }
