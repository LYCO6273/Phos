# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Phos 是一个基于计算光学概念的胶片模拟应用，通过计算光在底片上的行为来复现自然、柔美、立体的胶片质感。项目使用 OpenCV 进行图像处理，Streamlit 作为交互界面。

## 项目结构

- `Phos_0.1.1 copy.py` - 主要的 Streamlit 应用入口点（由 devcontainer 使用）
- `Phos_0.1.1.py` - 备用/较旧的版本
- `legacy/Phos_0.1.0.py` - 存档的历史版本
- `.streamlit/config.toml` - Streamlit 服务器配置（上传/消息大小限制为10MB）
- `.devcontainer/devcontainer.json` - Devcontainer 设置和默认运行命令

## 核心架构

### 图像处理流程（Phos_0.1.1 copy.py:480-525）

1. **图像标准化** (`standardize` 函数:176-202)
   - 将图像短边标准化为 3000 像素
   - 使用高质量插值方法

2. **光学建模** (`luminance` 函数:205-227)
   - 模拟不同乳剂层的分光吸收特性
   - 支持彩色和黑白两种模式

3. **胶片特性模拟** (`film_choose` 函数:57-173)
   - NC200: 彩色负片（灵感来自富士C200）
   - AS100: 黑白正片（灵感来自富士ACROS）
   - FS200: 高对比黑白正片

4. **光学扩散效应** (`opt` 函数:379-476)
   - 模拟光晕和散射效果
   - 根据平均亮度计算高光敏感度
   - 应用高斯模糊创建光晕层

5. **色调映射** (`filmic` 和 `reinhard` 函数)
   - Filmic: 基于胶片曲线的色调映射（版本 0.1.1 的改进）
   - Reinhard: 经典的全局色调映射算法

6. **颗粒效果** (`grain` 函数:238-309)
   - 基于加权随机噪声模拟胶片颗粒
   - 支持可调节强度（默认/柔和/较粗/不使用）
   - 彩色胶片考虑明度属性（版本 0.1.1 的改进）

## 开发命令

### 安装依赖
```bash
python -m pip install -r requirements.txt
```

### 运行应用
```bash
# 本地开发
streamlit run "Phos_0.1.1 copy.py"

# Devcontainer 环境（与配置匹配）
streamlit run "Phos_0.1.1 copy.py" --server.enableCORS false --server.enableXsrfProtection false
```

### 语法检查
```bash
python -m py_compile "Phos_0.1.1 copy.py"
```

## 依赖版本

- Python: 3.13（README）或 3.11（devcontainer）
- numpy: 2.2.6
- opencv-python-headless: 4.12.0.88
- streamlit: 1.51.0
- pillow: 12.0.0

注意：如果更新依赖或语言特性，需要同时更新 README.md 和 .devcontainer/devcontainer.json 中的版本要求。

## 编码规范

- Python 缩进：4个空格
- 函数命名：snake_case
- 常量命名：UPPER_SNAKE_CASE
- 将小的纯函数用于图像操作
- 将 UI 字符串和滑块默认值放在 Streamlit 布局代码附近

## 测试

目前没有配置自动化测试框架。如果添加测试，建议：
- 使用 `pytest`
- 放置在 `tests/` 目录（如 `tests/test_tonemapping.py`）
- 专注于确定性单元（色调映射、颗粒生成）
- 避免依赖 Streamlit 运行时

## 提交和 PR 指南

- 使用简洁的祈使句提交信息
- PR 应包括：简洁摘要、如何运行说明、UI/渲染变更的截图/GIF
- 避免意外提交如"Changes to be committed…"

## 安全与配置

- 不要在本地开发之外禁用 CORS/XSRF
- 如需修改 `.streamlit/config.toml`，请说明对上传和内存使用的影响
- 切勿提交 API 密钥、凭据或用于调试的私有图像

## 版本差异（0.1.0 → 0.1.1）

- 色调映射实现从 Reinhard 调整为 Filmic
- 彩色胶片颗粒实现考虑明度属性
- 主要变更在 `Phos_0.1.1 copy.py` 中

## 许可证

AGPL-3.0 - 详见 LICENSE 文件
