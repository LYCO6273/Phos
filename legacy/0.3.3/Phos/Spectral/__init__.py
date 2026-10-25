#赛博请神
# Spectral 包 —— 光谱迁移的共享地基
# 对外暴露：
#   migrate_rgb_primary() 输入端光谱迁移主入口（光谱重建，唯一路径）
#   migrate_bw_primary()  黑白单通道光谱迁移主入口

from .migrate import (
    migrate_bw_primary,
    migrate_rgb_primary,
)

__all__ = [
    "migrate_rgb_primary",
    "migrate_bw_primary",
]
