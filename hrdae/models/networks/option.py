from dataclasses import dataclass


@dataclass
class NetworkOption:
    activation: str = "sigmoid"  # "none" | "sigmoid" | "tanh" | "relu"
