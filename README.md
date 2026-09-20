# 语音转写工具（FunASR Paraformer）

基于 [FunASR](https://github.com/modelscope/FunASR) 的 Paraformer 中文语音转写脚本，带标点恢复、繁简转换、连续重复去除，输出 Markdown。

## 文件说明

```
whisper/
├── zhuan_paraform.py    # 主转写脚本
├── translate_srt.py     # SRT 字幕翻译（日文 → 简体中文，独立工具）
├── paraformer-zh/       # Paraformer 中文 ASR 模型（本地）
└── ct-punc/             # 标点恢复模型（ct-transformer 中英，本地）
```

## 环境准备

### 1. Python 依赖

```bash
pip install funasr modelscope torch
# 繁简转换（可选，三选一）：
pip install opencc-python-reimplemented   # 或 zhconv / hanziconv
```

### 2. FFmpeg

脚本通过 FFmpeg 提取音轨，确保 `ffmpeg` 在系统 PATH 中：

- Windows：`winget install Gyan.FFmpeg`
- macOS：`brew install ffmpeg`
- Linux：`sudo apt install ffmpeg`

## 用法

```bash
# 单文件转写（默认 CUDA）
python zhuan_paraform.py "D:\yt\video.mp4"

# 指定设备 / 输出目录 / 批大小（显存不足调低）
python zhuan_paraform.py "D:\yt\video.mp4" --device cpu --out-dir D:\out --batch-size-s 30
```

输出为 `.md` 文件（一级标题 + 正文），与源文件同名，重名时自动加 `(1)` 后缀。

## 说明

- 默认模型路径：`C:\whisper\paraformer-zh`（ASR）和 `C:\whisper\ct-punc\models\iic--punc_ct-transformer_cn-en-common-vocab471067-large\snapshots\master`（标点），可用 `--model-dir` / `--punc-model-dir` 覆盖。
- 模型文件（`.bin` / `.pt` 等）被 `.gitignore` 排除，不入库；本目录已预置模型，直接可用。
- GPU 加速：`--device cuda` 需 CUDA 环境；无 GPU 用 `--device cpu`。
