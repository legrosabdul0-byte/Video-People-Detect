<p align="right"><a href="README.md">English</a> | <b>中文</b></p>

# 视频人数检测 (Video People Detect)

使用 YOLOv8 检测器统计视频中的人数。针对教室这类**遮挡严重**的场景做了优化：
通过抽取多帧并对结果做聚合，让最终人数估计更稳健。

## 下载（Windows，无需安装 Python）

前往 [**Releases**](https://github.com/legrosabdul0-byte/Video-People-Detect/releases/latest)
页面下载最新的 `PeopleCounter-vX.Y.Z.exe`，双击即可运行。Python、所有依赖库以及
YOLO 权重都已经打包进这个 `.exe` 里，所以在从未装过 Python 的电脑上也能直接运行。

> 每次推送到 `main` 分支，GitHub Actions 都会自动构建一个新的 `.exe` 并发布到
> Releases，版本号从 **v1.0.0** 开始。

## 工作原理

1. 在视频中段按比例抽取多帧（可配置）。
2. 对抽取的帧批量运行 YOLO 人体检测（一次调用完成）。
3. 过滤掉过小的检测框（最小面积按分辨率自动缩放）。
4. 聚合各帧人数：先去掉最低的（很可能漏检的）几帧，再取较高的百分位数以补偿遮挡。
5. 根据各帧人数的相对离散程度计算置信度，并保存最接近最终人数那一帧的标注预览图。

## 安装（从源码运行）

```bash
pip install -r requirements.txt
```

首次运行时会自动下载 YOLO 权重（`yolov8s.pt`）。

## 使用方法

启动桌面图形界面：

```bash
python main.py
```

对单个视频进行无界面检测（直接打印结果，无需窗口）：

```bash
python main.py path/to/video.mp4
python main.py path/to/video.mp4 --no-preview
```

以代码方式调用检测器：

```python
from video_people_detect import PeopleDetector, DetectionConfig

detector = PeopleDetector(DetectionConfig(final_percentile=70))
result = detector.detect("classroom.mp4", log=print)
print(result.final_count, result.confidence)
```

## 配置

所有可调参数都集中在 [`video_people_detect/config.py`](video_people_detect/config.py)，
包括模型名称、置信度阈值、推理图像尺寸、抽帧位置、最小框面积比例、聚合百分位数、
是否批量推理以及运行设备。

## 项目结构

```
main.py                       # 图形界面 + 命令行 入口
video_people_detect/
    config.py                 # DetectionConfig（所有可调参数）
    detector.py               # PeopleDetector（与界面无关的核心逻辑）
    app.py                    # Tkinter 图形界面
.github/workflows/build-exe.yml   # CI：构建 Windows exe 并发布 Release
requirements.txt
```

## 相比原始单文件脚本的改进

- **模块化结构** —— 检测逻辑与 Tkinter 界面完全分离，可被命令行或测试复用。
- **跨平台预览** —— 用一个同时支持 macOS 和 Linux 的辅助函数替代仅限 Windows 的 `os.startfile`。
- **批量推理** —— 所有抽取的帧在一次 `predict` 调用中送入 YOLO，提升吞吐量（尤其是 GPU）。
- **自动选择设备** —— 有 CUDA 时优先使用，否则回退到 CPU。
- **无界面命令行模式** —— 无需打开窗口即可处理视频。
- **更安全的抽帧定位、类型注解和文档字符串**。
