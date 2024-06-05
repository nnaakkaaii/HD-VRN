import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ..transforms import Transform


@dataclass
class CTOption:
    pass


class CT(Dataset):
    pass
