#赛博请神
# Cache/stage_cache.py —— 0.3.4 阶段缓存（磁盘记忆化）
# 用途：把"只依赖参数/数据的纯阶段"按参数指纹缓存到磁盘，
#       改一个参数时其他阶段直接复用，方便单项修改的快速测试。
# 使用：cached("psf_fit", 2, {"film": "provia100f", "mtf_fp": "...", ...}, builder)
import hashlib
import json
import os
import pickle

import numpy as np

_CACHE_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "Data", "cache", "stage"))


def _ensure_dir():
    os.makedirs(_CACHE_DIR, exist_ok=True)


def key_hash(key_dict):
    """dict → 稳定哈希（排序 JSON）。"""
    payload = json.dumps(key_dict, sort_keys=True, ensure_ascii=False,
                         default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def cached(namespace, version, key_dict, builder):
    """按 (namespace, version, key_dict) 记忆化 builder() 的返回值。

    builder 的返回值会被 pickle 存入 npz；命中时直接反序列化返回。
    版本号或 key_dict 变化 → 自动失效重算。
    """
    _ensure_dir()
    path = os.path.join(_CACHE_DIR,
                        f"{namespace}_v{version}_{key_hash(key_dict)}.npz")
    if os.path.exists(path):
        try:
            with np.load(path, allow_pickle=True) as d:
                return pickle.loads(d["value"].item())
        except Exception:
            pass  # 损坏则重算
    value = builder()
    np.savez(path, value=np.frombuffer(pickle.dumps(value), dtype=np.uint8))
    return value
