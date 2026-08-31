# 你需要先安装依赖
#  python -m pip install faster-whisper pyannote.audio torch torchvision torchaudio


#  python duorenzhuanhuan.py "D:\test.mp4"


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
# 强制使用 torchaudio 后端，避免 torchcodec 错误
os.environ["PYANNOTE_AUDIO_BACKEND"] = "torchaudio"

import sys
import traceback
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

# ===== 模型目录 =====
WHISPER_MODEL_DIR = r"C:\whisper"

# ===== pyannote HuggingFace Token =====
# 去 https://huggingface.co/settings/tokens 创建，权限选 Read
HF_TOKEN = "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"   # 替换成你实际的 token


def get_available_filename(base_path: str) -> str:
    """如果文件存在，自动生成不重复文件名（name(1).txt）"""
    if not os.path.exists(base_path):
        return base_path
    dirname, basename = os.path.split(base_path)
    name, ext = os.path.splitext(basename)
    i = 1
    while True:
        new_name = f"{name}({i}){ext}"
        new_path = os.path.join(dirname, new_name)
        if not os.path.exists(new_path):
            return new_path
        i += 1


def format_time(seconds: float) -> str:
    """格式化时间（hh:mm:ss）"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    if len(sys.argv) < 2:
        print("用法：")
        print(r'python duorenzhuanhuan.py "D:\test.mp4"')
        sys.exit(1)

    input_file = sys.argv[1]

    if not os.path.exists(input_file):
        print(f"❌ 文件不存在: {input_file}")
        sys.exit(1)

    try:
        # ---------- 1. 验证 HuggingFace Token ----------
        print("🔑 正在验证 HuggingFace Token...")
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            user = api.whoami(token=HF_TOKEN)
            print(f"✅ Token 验证成功，登录用户：{user['name']}")
        except ImportError:
            print("⚠️ 未安装 huggingface_hub，跳过 Token 验证。如需验证请运行：pip install huggingface_hub")
        except Exception as e:
            print(f"❌ Token 无效或网络错误，请检查：{e}")
            sys.exit(1)

        # ---------- 2. 加载 faster-whisper 模型 ----------
        print("🚀 加载 faster-whisper 模型...")
        whisper_model = WhisperModel(
            WHISPER_MODEL_DIR,
            device="cuda",
            compute_type="float16"
        )

        # ---------- 3. 加载说话人识别模型（关键修改：use_auth_token -> token） ----------
        print("🚀 加载说话人识别模型（pyannote/speaker-diarization-3.1）...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HF_TOKEN      # 这里改成了 token
        )
        pipeline.to("cuda")

        # ---------- 4. Whisper 转写 ----------
        print("🎙️ 开始 Whisper 转写...")
        segments, info = whisper_model.transcribe(
            input_file,
            beam_size=5,
            language="zh"
        )

        whisper_segments = []
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            whisper_segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": text
            })

        # ---------- 5. 说话人识别 ----------
        print("👥 开始说话人识别...")
        diarization = pipeline(input_file)

        speaker_map = {}
        speaker_index = 1
        diarization_segments = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_map:
                speaker_map[speaker] = f"speaking{speaker_index}"
                speaker_index += 1
            diarization_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker_map[speaker]
            })

        # ---------- 6. 合并转写和说话人 ----------
        print("🔗 合并转写和说话人...")
        final_lines = []

        for ws in whisper_segments:
            ws_mid = (ws["start"] + ws["end"]) / 2
            matched_speaker = "unknown"
            for ds in diarization_segments:
                if ds["start"] <= ws_mid <= ds["end"]:
                    matched_speaker = ds["speaker"]
                    break
            start_time = format_time(ws["start"])
            line = f"[{start_time}] {matched_speaker}: {ws['text']}"
            final_lines.append(line)

        final_text = "\n".join(final_lines)

        # ---------- 7. 输出文件（自动避免重名） ----------
        output_file = os.path.splitext(input_file)[0] + "_多人转写.txt"
        output_file = get_available_filename(output_file)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_text)

        print()
        print("✅ 转写完成")
        print(f"📄 输出文件: {output_file}")
        print()
        print("👥 检测到的说话人：")
        for original, mapped in speaker_map.items():
            print(f"  {mapped} -> {original}")

    except Exception:
        print("❌ 出现异常：")
        traceback.print_exc()


if __name__ == "__main__":
    main()