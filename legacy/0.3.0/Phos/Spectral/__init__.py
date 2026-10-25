#赛博请神
# Spectral 包 —— 光谱迁移的共享地基
# 对外暴露：
#   migrate_rgb_primary() 输入端光谱迁移主入口（CIE-RGB 色匹配函数伪发射）

from .migrate import migrate_rgb_primary

__all__ = ["migrate_rgb_primary"]
