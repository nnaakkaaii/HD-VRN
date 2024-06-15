from typing import Protocol


class LRScheduler(Protocol):
    def step(self, epoch: int | None = None) -> None: ...

    def get_last_lr(self) -> list[float]: ...
