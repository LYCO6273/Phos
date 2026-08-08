# Phos 0.3.0 — 输入端光谱迁移验证版

0.3.0 的目标只有一个：验证“把输入重建为光谱，再喂给各乳剂层”这条路能不能走通。
因此输入输出架构全部重做，但胶片端的影调与颗粒尽量沿用 0.2.3。

## 运行

```bash
cd Phos/0.3.0/Phos
pip install -r requirements.txt
streamlit run main.py
```

自检（不依赖 streamlit）：

```bash
cd Phos/0.3.0
python tools/selftest.py
```

## 上传报错 “AxiosError: Request failed with status code 400”

这个 400 来自 Streamlit 的上传接口，发生在进入 Phos 代码之前，常见原因与解决：

1. **文件超过上传上限**：仓库根目录的 `.streamlit/config.toml` 将上限设为 50MB，
   iPhone 48MP ProRAW DNG 很容易超过。0.3.0 自带的
   `Phos/.streamlit/config.toml` 已把上限提到 500MB，请从
   `Phos/0.3.0/Phos` 目录启动 `streamlit run main.py`（或在
   `Phos/0.3.0/Phos/.streamlit/config.toml` 中继续调大）。
2. **tornado 版本过新**：tornado 6.5.x 有上传 400 的回归，请安装
   `pip install tornado==6.4.1`（已写入 requirements.txt）。
3. **文件名含中文/Emoji**：部分 Streamlit 版本对非 ASCII 文件名返回 400，
   先改成纯英文文件名（如 `IMG_0001.dng`）即可确认。
4. **代理或内嵌浏览器**：如果通过代理/内嵌 WebView 访问，XSRF 头可能丢失；
   改用本机普通浏览器（Chrome/Safari）访问 `http://localhost:8501`。

## 管线

```text
RAW/JPEG
  -> 线性 XYZ(D65) 浮点（rawpy 相机矩阵 + 相机白平衡，保留 >1 高光）
  -> xyz->反射率 LUT（离线 QP：min rᵀQr  s.t. Mr=c, 0≤r≤1）
  -> 乳剂层曝光 LUT（对每款胶片：E_i = ∫ S_i(λ)·D65(λ)·r(λ)dλ）
  -> 逐像素一次三维插值，得到各层曝光量
  -> 光学扩散 + 颗粒
  -> HP5：单层特性曲线；Gold/Vision：特性曲线 -> C/M/Y 染料光谱堆叠
     -> 扫描 -> 负片反转 -> 灰场/黑场校准 -> sRGB
```

光谱数据：CIE 1931 2°（5nm，由 1nm 带宽平均）、D65（5nm），来源为 colour-science
开源数据（CIE 15:2004/ASTM E308）。`M` 矩阵按行归一化，使理想白板精确映射到 sRGB
D65 白点 (0.95047, 1.0, 1.08883)。

## 各胶片的数据来源与近似

- **HP5**：特性曲线从官方 PDF 数字化（48 点）。光谱灵敏度只有定性楔形图，因此用
  CIE 光度函数 ȳ(λ) 近似全色响应，即 HP5 曝光量 = 场景亮度 Y。
- **Gold 200**：特性曲线与 D-min（橙色色罩）来自官方 PDF。光谱灵敏度在 PDF 里以
  离散竖线存储、难以可靠重建，0.3 暂用 Vision 200T 灵敏度作代理（已注释说明）；
  C/M/Y 染料光谱也用 Vision 200T（同为柯达彩负染料家族）。
- **Vision 200T**：特性曲线、光谱灵敏度、C/M/Y 染料光谱、D-min 全部来自官方
  中文数据表，是 0.3 数据最完整的胶片。
- **Provia 100F**：曲线页为位图，0.3 不纳入。

数据表数字化脚本：`tools/extract_datasheets.py`（需要 `pdfplumber`，仅开发用）。

## 已知说明与限制

- **iPhone DNG 的 `bright=3.0`**：iPhone ProRAW/DNG 元数据带 BaselineExposure
  补偿（约 +1~2 EV），rawpy 的 `no_auto_bright` 不读它，所以不加增益会明显偏暗。
  `bright=3.0` 是对该补偿的简化替代，保留 0.2.3 行为。
- **高光策略**：沿用 rawpy 16bit 输出，高光保留到约 3~4× 白电平，超出裁剪；
  颗粒步骤仍会像 0.2.3 一样把层曝光限到 [0,1]。
- **光谱重建是建模选择**：XYZ 只能确定同色异谱类，光滑先验给出其中一个确定性解；
  对自发光/屏幕内容，“反射率”模型并不严格成立。
- **灰场/黑场校准**：染料串扰和色罩在扫描 RGB 密度域留下残差，0.3 用
  “黑场→0、18% 灰→中灰”的两点线性校准模拟扫描仪标定。
- **性能**：LUT 首次构建约 3 秒（缓存在 `Data/cache/`）；3024×3024 全尺寸下
  HP5 约 1~2 秒，Gold 约 23 秒，Vision 约 31 秒（大头是 61 波段逐像素计算）。
