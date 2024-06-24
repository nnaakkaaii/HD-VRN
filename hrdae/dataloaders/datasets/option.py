from dataclasses import dataclass


@dataclass
class DatasetOption:
    wrap: str = "none"  # none, divide, serialize
