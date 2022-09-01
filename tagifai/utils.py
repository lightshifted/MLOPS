import json
import random
from typing import Dict, List, Tuple

import numpy as np


def load_dict(filepath: str) -> Dict:
    """Load a dictionary from a JSON's filepath.

    Args:
        filepath (str): _description_

    Returns:
        Dict: _description_
    """
    with open(filepath, "r") as fp:
        d = json.load(fp)
    return d


def save_dict(d: Dict, filepath: str, cls=None, sortkeys: bool = False) -> None:
    """Save a dictionary to a specific location.

    Args:
        d (Dict): _description_
        filepath (str): _description_
        cls (_type_, optional): _description_. Defaults to None.
        sortkeys (bool, optional): _description_. Defaults to False.
    """
    with open(filepath, "w") as fp:
        json.dump(d, indent=2, fp=fp, cls=cls, sort_keys=sortkeys)


def set_seeds(seed: int = 42) -> None:
    """Set seed for reproducibility.

    Args:
        seed (int, optional): _description_. Defaults to 42.
    """
    # Set seeds
    np.random.seed(seed)
    random.seed(seed)
