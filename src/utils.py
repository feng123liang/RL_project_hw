from __future__ import annotations

import json
import os
import random
from typing import Dict, Iterable, List

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def moving_average(values: Iterable[float], window: int = 50) -> List[float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return []
    if arr.size < window:
        return arr.tolist()
    kernel = np.ones(window, dtype=np.float64) / window
    smooth = np.convolve(arr, kernel, mode="valid")
    prefix = [float(smooth[0])] * (window - 1)
    return prefix + smooth.tolist()


def save_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
