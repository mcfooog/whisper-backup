# Whisper 转写工具集

基于 [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 的音视频转写 / 翻译 / 字幕生成工具集，支持中文、英文、日语、俄语等多种语言的转写与互译，并提供多人说话人识别、文件夹批量处理等功能。

## 功能一览

| 类别 | 脚本 | 说明 |
| --- | --- | --- |
| 单文件转写 | `zhuanwenzi.py` | 核心转写脚本（中文），含繁简转换，被批量脚本调用 |
| 单文件转写 | `zhuan_en.py` | 英文转写，输出 `_en.txt` |
| 单文件字幕 | `zimu.py` | 中文 SRT 字幕生成，输出 `_zh.srt` |
| 单文件字幕 | `zimu_en.py` | 英文 SRT 字幕生成，输出 `_en.srt` |
| 多语转写 | `zhuan_jp.py` | 日语转写，输出 `_jp.srt` |
| 多语转写 | `zhuan_eyu.py` | 俄语转写并翻译为中文，输出 `_eyu_cn.txt` |
| 多人转写 | `duorenzhuanhuan.py` | 基于 pyannote 的说话人识别转写，输出 `_多人转写.txt` |
| 文件夹批量 | `zhuanhuanwenjianjia.py` | 批量扫描文件夹媒体文件并调用 `zhuanwenzi.py` |
| 文件夹批量 | `zhuanwenjianjia_en.py` | 批量英文转写（调用 `zhuan_en.py`） |
| 断点续跑 | `duandianzhuanhuan.py` | 带断点/重试的批量转写 |
| 字幕翻译 | `translate_srt.py` | 日文 SRT 翻译为简体中文，输出 `_cn.srt` |

## 目录结构

```
whisper/
├── zhuanwenzi.py            # 核心中文转写
├── zhuan_en.py              # 英文转写
├── zimu.py / zimu_en.py     # 中/英文字幕生成
├── zhuan_jp.py / zhuan_eyu.py   # 日语/俄语转写
├── duorenzhuanhuan.py       # 多人说话人识别
├── zhuanhuanwenjianjia.py       # 文件夹批量（中文）
├── zhuanwenjianjia_en.py        # 文件夹批量（英文）
├── duandianzhuanhuan.py     # 断点续跑批量
├── translate_srt.py         # SRT 字幕翻译
├── model.bin                # ⚠️ Whisper 模型权重（约 2.9 GB，未入库，见下）
├── config.json              # Whisper 模型配置
├── tokenizer.json           # 分词器
├── preprocessor_config.json # 预处理器配置
├── vocabulary.json          # 词表
└── .gitignore
```

## 环境准备

### 1. 安装 Python 依赖

```bash
pip install faster-whisper tqdm deep-translator
# 多人说话人识别额外需要：
pip install pyannote.audio torch torchaudio
# 繁简转换（可选，三选一）：
pip install opencc-python-reimplemented   # 或 zhconv / hanziconv
```

### 2. 安装 FFmpeg

脚本通过 FFmpeg 提取音轨，请确保 `ffmpeg` 在系统 PATH 中。

- Windows：`winget install Gyan.FFmpeg` 或从 [ffmpeg.org](https://ffmpeg.org/download.html) 下载
- macOS：`brew install ffmpeg`
- Linux：`sudo apt install ffmpeg`

### 3. 获取模型权重

`model.bin`（约 2.9 GB）**未随仓库上传**，需自行下载后放入项目根目录，与 `config.json`、`tokenizer.json` 等放在一起：

```
c:\whisper\model.bin
c:\whisper\config.json
c:\whisper\tokenizer.json
...
```

可从 [HuggingFace](https://huggingface.co/SYSTRAN) 或 faster-whisper 官方渠道获取对应的 Whisper 模型权重。

## 用法示例

```bash
# 单文件中文转写
python zhuanwenzi.py "D:\yt\video.mp4" --model-dir C:\whisper --device cuda

# 单文件英文字幕
python zimu_en.py "D:\yt\video.mp4" --model-dir C:\whisper

# 文件夹批量转写（自动识别视频/音频）
python zhuanhuanwenjianjia.py "D:\yt"

# 多人说话人识别
python duorenzhuanhuan.py "D:\test.mp4"

# 日文 SRT 翻译成中文
python translate_srt.py "input.srt"
```

> 批量脚本支持的媒体格式见 [zhuanhuanwenjianjia.py](zhuanhuanwenjianjia.py) 中的 `MEDIA_EXTENSIONS`，涵盖常见视频（mp4/mkv/avi/mov/ts/webm/rmvb 等）与音频（mp3/wav/m4a/aac/flac/ogg/opus/wma 等）。

## 说明

- GPU 加速：`--device cuda` 需 CUDA 环境；无 GPU 时用 `--device cpu`。
- 翻译类脚本（`zhuan_eyu.py`、`translate_srt.py`）调用 Google 翻译，需联网。
- `duorenzhuanhuan.py` 的 pyannote 模型需配置 `HUGGING_FACE_HUB_TOKEN`。

## 许可

脚本仅供学习与个人使用，模型权重遵循各自原始许可。
