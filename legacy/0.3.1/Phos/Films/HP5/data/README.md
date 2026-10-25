# HP5 Plus（ILFORD，ISO 400 黑白负片）模型数据 —— 0.3.1 重构

构建依据：`Films/STANDARD_FLOW.md` 第二部分（黑白负片标准流程）。
数据来源：`胶片技术文档/Ilford/黑白/HP5 Plus` + `Kodak/黑白/TRI-X 400`。

## 数据清单

- char_curves_hp5.csv/npz —— 感光特性曲线（密度 vs 绝对 logH，ILFOTEC HC 显影）。
  **0.2.3 版本手描 21 控制点**（从 HP5 技术文档扒取）PCHIP 插值外推到均匀网格
  （相对轴 0–5，步长 0.05，101 点）。相对轴灰点 1.9896 → 按 ISO 400 标定平移
  -3.49475，logH_gray = log10(12.5/400) = -1.50515。
- granularity_hp5.csv/npz —— 漫射 rms 颗粒度（48µm 环形孔径），X 轴=净密度
  （0.25–2.90，步长 0.05），rms 峰值 17@D=1.0。黑白模型在密度域直接查表。
- spectral_mtf_hp5.npz —— MTF：**KODAK TRI-X 400 代理**（D-76 显影，
  diffuse visual MTF）。HP5 文档无 MTF 曲线，按同级 ISO 400 黑白卷代理；
  22 个描点（2–75 cyc/mm），稠密网格 0.5–200 cyc/mm，高频按对数线性衰减外推。
- spectral_sensitivity_hp5.csv —— 光谱灵敏度（352-660nm 单列描点），
  供 Spectral/migrate.py 的 migrate_bw_primary() 单通道光谱迁移使用。

## 标定记录

- 曝光标定：K=12.5、EI=400 → logH_gray=-1.50515（2026-08-28）
- 黑白反转锚点：黑端 D=0.175（0,0,0 输入 → D_min），
  白端 D=1.270218（1,1,1 输入）；灰点（0.18）D≈0.7898
- 输出伽马：OUT_GAMMA=1.4
- 光谱输入：CIE-RGB 色匹配函数伪发射 → 实测灵敏度积分（白场 E=1），
  权重 R/G/B ≈ 0.328/0.354/0.318（全色响应近等权）

## 待办

- HP5 官方 MTF 若后续找到，替换 Tri-X 400 代理
- 颗粒度 npz 的早期"500T 形状代理"备注已作废，现为实测描点
