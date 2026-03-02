import os
import yaml
import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self):
        return self.sum / max(1, self.count)

    def update(self, value, n=1):
        self.sum += value * n
        self.count += n


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_checkpoint(state, path):
    torch.save(state, path)
