from dataclasses import dataclass


@dataclass
class DataLoaderOption:
    batch_size: int = 32
    train_val_ratio: float = 0.8
