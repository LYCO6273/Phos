#赛博请神
# Input 包 —— 第一段（原片解析）：原片 → 线性光照度
# 对外暴露统一入口 decode_to_lux()；JPEG/RAW 的具体解码
# 分别在 Input/jpeg.py 与 Input/raw.py 中实现。

from .decode import decode_to_lux

__all__ = ["decode_to_lux"]
