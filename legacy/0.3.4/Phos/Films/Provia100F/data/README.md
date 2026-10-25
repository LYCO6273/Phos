# Provia 100F（富士反转片）备用数据

用途：彩色正片模型（Provia 100F 蓝本）构建储备。目前仅有特性曲线，
正片完整流程见 STANDARD_FLOW.md 第三部分（待整理）。

## 数据清单

- Provia100F_characteristic_curve.csv —— 0.2.2 PV100.py 手描描点
  （来自 Provia 100F 技术文档），三层（R/G/B 感光层）负斜率正像曲线。
- Provia100F_characteristic_curve_interp.csv —— PCHIP 插值网格
  （绝对 logH 轴，步长 0.05，81 点；ISO 100 标定，见下）。
- Provia100F_characteristic_curve.npz —— 运行时数据源：
  log10_exposure_lux_seconds + density_R/G/B + logH_gray/x_gray。
  曝光标定：logH_gray=log10(12.5/100)=-0.90309，灰点取三层 D=0.9
  （正片中灰）处相对轴 x=2.305 → 轴平移 -3.20809。旧模型灰点 1.3088
  已弃用（其把 18% 灰放在 D≈2.9，不符正片中灰）。
- Provia100F_spectral_sensitivity.csv/npz —— 吸收光谱（log 灵敏度描点）：
  对数域 PCHIP 插值到 400-700nm（61 点，5nm）+ 端点斜率外推，
  log 下限 -4.0；lin = 10^log（保持三层相对比例，峰值蓝 12.2/绿 6.5/红 10.2）。
- Provia100F_dye_density.csv/npz —— 染料光谱密度（漫射光谱密度描点）：
  PCHIP 插值到 400-700nm（61 点）；存原始密度 + 峰值归一化 norm。
  峰值：黄 0.998@445nm / 品红 0.995@544nm / 青 0.997@659nm。
- Provia100F_MTF.csv/npz —— MTF（17 点描点）：PCHIP 插值稠密网格
  （0.5-200 cyc/mm，128 点）+ 高频对数衰减外推；低频平台 ~114%（邻界）。
- Provia100F_granularity.csv/npz —— 颗粒度：**5213 形状代理 × 各层放缩**
  （R/G/B = 1.49/0.92/0.42），锚定 datasheet 的 DIFFUSE RMS GRANULARITY VALUE=8
  （48µm 孔径，D=1.0）；logE 轴平移到 Provia 灰点（ISO 100）。

## 曲线特征

- 负斜率：低曝光 → 高密度（D_max ≈ 3.3-3.43），高曝光 → 低密度（D_min ≈ 0.09）
- 正像：扫描透光率 T=10^(-D) 直接是正像亮度，输出端无反转
- 层映射沿用彩负约定：R=红感光/青染料、G=绿感光/品红、B=蓝感光/黄
- 18% 灰（0.18 线性输入）→ 三层 D≈0.90（中灰）

## 待补数据（构建正片模型时需要）

- 染料中灰曲线 + D-min（三层染料已有；正片 D-min 接近中性，可近似或待补）
- 颗粒度代理说明：R 层峰值 rms≈14.6 偏粗（继承 5213 形状），
  若实拍验证后需修正，可调整放缩或换成更接近正片的形状
- 输出端 gamma 语义确认（建议 sRGB 显示编码 1/2.2，而非负片风格化 gamma）

注：本数据存于项目内备用；正式归档（胶片技术文档/Fujifilm/正片/Provia 100F/）
需在目录可写时同步一份。
