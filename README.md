# Phos - 基于计算光学的胶片模拟

# 综述General

  你说的对，但是 Phos. 是基于「计算光学」概念的胶片模拟。通过计算光在底片上的行为，复现自然、柔美、立体的胶片质感。
  
  Hello! Phos. is a film simulation app based on the idea of "Computational optical imaging“. By calculating the optical effects on the film,we could recurrent the natural, soft, and elegant tone of these classical films.
  
  这是一个原理验证demo，图像处理部分基于opencv，交互基于streamlit平台制作，部分代码使用了AI辅助生成。
  
  This is a demo for idea testing. The image processing part is based on OpenCV, and the interaction is built on the Streamlit platform. Some of the code was generated with the assistance of AI.

  如果您发现了项目中的问题，或是有更好的想法想要分享，还请通过邮箱 lyco_p@163.com 与我联系，我将不胜感激。
  
  If you find any issues in the project or have better ideas you would like to share, please contact me via email at lyco_p@163.com. I would be very grateful.

# 依赖Requirments

本项目基于Python 3.13编写

This project is based on Python 3.13

本项目依赖以下库：

This project requires these packages below:

numpy                     2.2.6
opencv-python             4.12.0.88
streamlit                 1.51.0
pillow                    12.0.0
rawpy                     (optional, for RAW)

兼容性尚不明确，如果运行出现问题，请以此处标明的依赖为准。

Compatibility is not yet clear. If any issues occur during operation, please refer to the dependencies listed here.

RAW 支持说明：
- 解析相机 RAW（如 DNG/NEF/CR2/ARW 等）依赖 `rawpy`（底层需要 libraw；在部分平台可能需要编译环境或系统依赖），可选安装：`python -m pip install -r requirements-raw.txt`。
- README 里标注的 Python 为 3.13，但 devcontainer 目前使用 Python 3.11；如遇安装/兼容问题请优先以 devcontainer 环境为准。

# 运行方式

## Streamlit（legacy UI）

`streamlit run "Phos_0.1.1 copy.py"`

## 前后端分离（FastAPI + 独立前端）

后端依赖：`python -m pip install -r requirements-api.txt`

启动后端：`uvicorn backend.main:app --host 0.0.0.0 --port 7000`

打开前端：浏览器访问 `http://localhost:7000/`（静态前端在 `frontend/`，通过 FastAPI 静态托管；核心处理在 `phos/`）。

# 许可证License

本项目采用 AGPL-3.0 许可证。

This project is licensed under AGPL-3.0.

你可以：

You may:

自由使用、研究、修改源代码

Freely use, study, and modify the source code

用于个人或教育项目

Use for personal or educational projects

用于开源项目（同样遵循AGPL）

Use for open source projects (also following AGPL)

你必须：

You must:

公开任何基于本项目的修改代码

Publicly release any modified code based on this project

保留原作者版权声明

Preserve the original author's copyright notice

同样采用AGPL许可证分发衍生作品

Distribute derivative works under the same AGPL license

商业使用请联系作者获取授权。

For commercial use, please contact the author for authorization.

完整许可证条款见 LICENSE 文件。

Full license terms are available in the LICENSE file.

# 作者Author

由@LYCO6273开发

developed by @LYCO6273

https://github.com/LYCO6273/Phos

联系方式：lyco_p@163.com

contact at: lyco_p@163.com
