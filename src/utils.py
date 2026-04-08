"""Utilities: seed, metrics, early stopping, result saving."""

import json
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 2021) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


class EarlyStopping:
    def __init__(self, patience: int = 10, delta: float = 0.0, save_path: str = "checkpoint.pt"):
        self.patience = patience
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss: Optional[float] = None
        self.should_stop = False

    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def load_best(self, model: torch.nn.Module) -> None:
        model.load_state_dict(torch.load(self.save_path, weights_only=True))


def save_results(results: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")


class Timer:
    def __init__(self):
        self.start_time = None

    def start(self) -> None:
        self.start_time = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start_time
