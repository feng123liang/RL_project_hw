from __future__ import annotations

"""通用工具函数。

该文件提供训练/评估流程都会复用的小工具，包括：
- 随机种子设置
- 目录创建
- 滑动平均计算
- JSON 序列化保存
"""

import json
import os
import random
from typing import Dict, Iterable, List

import numpy as np


def seed_everything(seed: int) -> None:
    """统一设置 Python 与 NumPy 的随机种子，保证结果可复现。"""

    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str) -> None:
    """若目录不存在则创建。"""

    os.makedirs(path, exist_ok=True)


def moving_average(values: Iterable[float], window: int = 50) -> List[float]:
    """计算一维序列的滑动平均，用于平滑奖励曲线。"""

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
    """将字典以可读格式写入 JSON 文件。"""

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
