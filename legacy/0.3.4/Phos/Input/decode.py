#赛博请神
# Input/decode.py —— 第一段（原片解析）统一入口
# 职责：把任意受支持的原片字节流解析成"线性光照度"三通道，
#       作为第二段（胶片模拟）的标准输入。
# 契约：
#   输入  file_bytes（原片字节）+ file_name（用于判断格式）
#   输出  (lux_r, lux_g, lux_b)：float32 二维数组，线性光照度，
#         遵循统一输入契约：相对中灰 E_mid（中灰 = 1.0），允许 >1。
#         JPEG：像素/255 + sRGB EOTF（数转胶，白场=1.0）；
#         DNG：rawpy 线性解码（bright=1.0 相对饱和）÷0.1282 × 2^BE
#         （ISO 12232：H_gray/H_sat=10/78≈0.1282 → E_mid=1；再乘
#         BaselineExposure 补偿
#         相机自己的中灰定义，使"相机认为的正确曝光"恒落在 E_mid=1）。
import io
import os


class _NamedBytes(io.BytesIO):
    """给内存字节流补一个 .name，兼容 Input.jpeg / Input.raw 的读取接口。"""
    def __init__(self, data, name):
        super().__init__(data)
        self.name = name


def decode_to_lux(file_bytes, file_name, optical=True):
    """原片字节 → 线性光照度（第一段主入口）。"""
    uploaded = _NamedBytes(file_bytes, file_name)
    extension = os.path.splitext(file_name)[1].lower()
    if extension in (".jpg", ".jpeg", ".png"):
        from Input.jpeg import jpeg_to_lux
        return jpeg_to_lux(uploaded)
    if extension == ".dng":
        from Input.raw import raw_to_lux
        return raw_to_lux(uploaded, optical=optical)
    raise RuntimeError("不支持的文件格式，请上传JPEG/PNG或DNG图像。")
