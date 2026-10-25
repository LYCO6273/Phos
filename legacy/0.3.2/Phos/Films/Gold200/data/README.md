# Gold 200（柯达 Gold 200，E-7022）模型数据 —— 0.3.1 重制用
数据来源：KODAK Gold 200 Technical Data E-7022（2016，Kodak Alaris，4 页）。
结构对齐 5213：raw + 插值，供 Gold200.py 按 5213 流程消费。

## 已数字化（本数据目录自产）
- char_curves_gold200.csv/npz —— 感光特性曲线（B/G/R 密度 vs logE，-3.0..0.9，PCHIP 保型插值）。
  logE=-1.0 时 D≈R0.98/G1.46/B1.81；logE=0 时 D≈R1.53/G2.01/B2.42。
  注：datasheet 标注 "Log H Ref: -1.14"；真实曝光标定（18% 灰落点）在建模时确定。
  注：logE 轴已微调为严格均匀网格（步长 0.05，float64 存储），模型查表走
  免二分快速插值；以后新胶片的特性曲线/颗粒度曲线同样统一插值到均匀网格。
- spectral_dye_density_gold200.csv/npz —— 光谱染料密度（400-700，5nm）。
  中灰/最低为 Gold 实测（用户描点）；逐染料 = 5213(ECN-2) 形状 + 数学修正：
  ① 幅度拟合 → ② 逐波长残差按各染料贡献占比回填并 Savitzky-Golay 平滑。
  修正后重建中灰相对残差约 0.7%（修正前约 7%）。
  峰位：黄 450nm / 品红 535nm / 青 690nm（青峰因 Gold 深红区净密度高而右移）。

## 代理（缺失内容，按"入门彩负/相似"原则）
- spectral_mtf_gold200.csv/npz —— MTF 用富士 Superia 200（入门彩负代表）用户描点，
  PCHIP 保型插值到 5213 频率网格；无分层，R=G=B 三层共用。低频平台 127%（邻界效应）。
- granularity_gold200.csv/npz —— Gold 无 rms 曲线；形状用 5213，放缩系数 1.9（估测，
  因 Gold 为入门消费彩负、颗粒明显更粗）。待 C200 实测修正。
  logE 轴同步微调为严格均匀网格（步长 0.05），与特性曲线共用快速查表。
- spectral_sensitivity_gold200.csv/npz —— Gold 光谱灵敏度（已补充，用户手工描点）。
  log=感光对数值，lin=10^log；黄层=蓝感光(blue) 峰@470nm，品红层=绿感光(green) 峰@549nm，
  青层=红感光(red) 峰@655nm。PCHIP 保型插值 400-700 5nm。
  建模时可在 Spectral/migrate.py 把 gold200 指向此数据，替换 5213 代理。

## C200 实测参考（桌面 /C200样张）
- 平场 L 通道颗粒 σ ≈ 1.2~6.9（0-255），中位约 2.5。
  说明 Gold 类消费卷颗粒显著；放缩系数 1.9 为初值，最终应以渲染观感/密度域闭环标定。
