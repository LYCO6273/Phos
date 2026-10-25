#赛博请神
# migrate.py —— 输入端光谱迁移（共享地基，唯一路径：光谱重建）
# 作用：把线性 RGB 升级为"到达胶片的光谱 → 按乳剂层灵敏度积分"，
#       输出仍是三个浮点通道（层曝光量），量纲与输入一致
#       （白场≈1，高光>1 按比例保留）。
# 实现说明：
#   0.3.2 起光谱重建（reconstruct.py）与 CMF 伪发射做过 A/B 实验，
#   光谱重建在纯红偏橙修正、暗部行为上明显更好，且物理上更接近
#   "模拟拍摄"假设；0.3.3 起 CMF 伪发射路径整体移除，这里只是
#   reconstruct 的对外统一入口，各胶片无需关心内部实现。

from .reconstruct import (
    migrate_bw_reconstruct,
    migrate_rgb_reconstruct,
)


def migrate_rgb_primary(lux_r, lux_g, lux_b, film="5213"):
    """光谱重建 RGB→三层曝光（彩负/正片）。

    参数：
        lux_r/lux_g/lux_b：线性 RGB 浮点通道（同形状二维数组）。
        film：胶片名，支持 "hp5" / "gold200" / "5213" / "provia100f"。
    返回：
        (out_r, out_g, out_b)：三层曝光量浮点通道（R/G/B 顺序），
        白场（1,1,1）→ 每层曝光 ≈1。
    """
    return migrate_rgb_reconstruct(lux_r, lux_g, lux_b, film)


def migrate_bw_primary(lux_r, lux_g, lux_b, film="hp5"):
    """光谱重建 RGB→单通道曝光（黑白）。

    参数：
        lux_r/lux_g/lux_b：线性 RGB 浮点通道（同形状二维数组）。
        film：黑白胶片名，目前支持 "hp5"。
    返回：
        单通道曝光量浮点数组（H,W），白场（1,1,1）→ 曝光 ≈1。
    """
    return migrate_bw_reconstruct(lux_r, lux_g, lux_b, film)
