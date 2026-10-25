#赛博请神
# Scanner 包 —— 第三段（独立扫描仪模块）：扫描 → 成片
# 对外暴露 render() 与 FILM_PROFILES；数值与 0.3.2 各胶片内部
# 反转/显示逻辑一致（0.3.3 起统一收敛到此处）。

from .scanner import FILM_PROFILES, SCANNER_OPTIONS, get_profile, render

__all__ = ["FILM_PROFILES", "SCANNER_OPTIONS", "get_profile", "render"]
