import numpy as np


def seed_everything(seed=42):

    import random
    import torch

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


def normalize_array(arr):

    arr = np.array(arr)

    return (arr - arr.mean()) / arr.std()