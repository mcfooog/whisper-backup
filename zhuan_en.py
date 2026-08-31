# python zhuan_en.py "D:\yt\video.mp4"

import os
import sys
import re
import argparse
import traceback

DEFAULT_MODEL_DIR = r"C:\whisper"


def get_available_filename(base_path: str) -> str:
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


def clean_common_filler(text: str) -> str:
    text = re.sub(r'^(?:\s*(?:uh|um|ah|eh|erm|ahh|uhh|…|\.|\s)+\s*){2,}$',
                  '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'(\b(?:uh|um|ah|eh|erm)\b)(?:[\s.…·，,]*){2,}',
                  ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def split_sentences(text: str) -> list:
    if not text.strip():
        return []
    delimiter_pattern = r'(?<=[。！？；!?;])'
    sentences = re.split(delimiter_pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= 1 and ' ' in text:
        sentences = [seg.strip() for seg in text.split(' ') if seg.strip()]
    return sentences


def normalize_sentence(s: str) -> str:
    s = s.strip()
    s = re.sub(r'[。！？!?]+$', '', s)
    return s


def remove_consecutive_duplicates(sentences: list) -> list:
    if not sentences:
        return sentences
    cleaned = [sentences[0]]
    last_norm = normalize_sentence(sentences[0])
    for i in range(1, len(sentences)):
        curr_norm = normalize_sentence(sentences[i])
        if curr_norm != last_norm:
            cleaned.append(sentences[i])
            last_norm = curr_norm
    return cleaned


def transcribe_file(
    input_path: str,
    model_dir: str,
    device: str = "cuda",
    compute_type: str = "float16",
    beam_size: int = 5,
    out_dir: str = None,
    verbose: bool = False,
    remove_filler: bool = False,
):
    try:
        from faster_whisper import WhisperModel
        from tqdm import tqdm
    except Exception as e:
        raise RuntimeError(
            "无法导入 faster_whisper 或 tqdm。请确认已安装：pip install faster-whisper tqdm"
        ) from e

    print(f"🚀 正在加载模型: {model_dir} (device={device}, compute_type={compute_type}) ...", flush=True)
    try:
        model = WhisperModel(model_dir, device=device, compute_type=compute_type)
    except Exception as e:
        print("❌ 模型加载失败（请确认 model_dir、CUDA 驱动与显存是否正常）。")
        raise

    print(f"🎙️ 开始转写英语: {input_path}", flush=True)

    transcribe_kwargs = {
        "beam_size": beam_size,
        "language": "en",
        "vad_filter": True,
        "condition_on_previous_text": False,
    }
    try:
        transcribe_kwargs["repetition_penalty"] = 1.1
        transcribe_kwargs["no_repeat_ngram_size"] = 4
        transcribe_kwargs["temperature"] = 0.0
        segments, info = model.transcribe(input_path, **transcribe_kwargs)
    except TypeError:
        transcribe_kwargs.pop("repetition_penalty", None)
        transcribe_kwargs.pop("no_repeat_ngram_size", None)
        transcribe_kwargs.pop("temperature", None)
        segments, info = model.transcribe(input_path, **transcribe_kwargs)
    except Exception:
        segments, info = model.transcribe(
            input_path,
            beam_size=beam_size,
            language="en",
            condition_on_previous_text=False
        )

    total_duration = round(info.duration, 2)
    pbar = tqdm(total=total_duration, unit="s", desc="转写进度",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    last_pos = 0
    raw_texts = []

    for seg in segments:
        delta = seg.end - last_pos
        if delta > 0:
            pbar.update(delta)
        last_pos = seg.end
        t = getattr(seg, "text", None)
        if t is None:
            t = str(seg)
        t = t.strip()
        if t:
            raw_texts.append(t)
        if verbose:
            preview = t.replace("\n", " ")
            if len(preview) > 50:
                preview = preview[:50] + "..."
            pbar.write(f"[片段] {preview}")
    pbar.close()

    full_text = " ".join(raw_texts).strip()

    if remove_filler:
        full_text = clean_common_filler(full_text)

    sentences = split_sentences(full_text)
    original_len = len(sentences)
    sentences = remove_consecutive_duplicates(sentences)
    removed = original_len - len(sentences)
    if removed > 0:
        print(f"🧹 后处理去除了 {removed} 句连续重复的内容", flush=True)

    output_text = "\n".join(sentences)
    output_text = re.sub(r'\n\s*\n+', '\n', output_text).strip()
    if output_text and not output_text.endswith('\n'):
        output_text += '\n'

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_path))[0] + "_en.txt"
        out_file_candidate = os.path.join(out_dir, base_name)
    else:
        out_file_candidate = os.path.splitext(input_path)[0] + "_en.txt"

    out_file = get_available_filename(out_file_candidate)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(output_text)

    try:
        lang = getattr(info, "language", None)
        prob = getattr(info, "language_probability", None)
        if lang:
            print(f"🌍 检测语言: {lang} (概率 {prob:.2f})", flush=True)
    except Exception:
        pass

    print(f"✅ 转写完成，已保存: {os.path.abspath(out_file)}", flush=True)
    return os.path.abspath(out_file)


def main():
    parser = argparse.ArgumentParser(description="基于 faster-whisper 的英语纯文本转写")
    parser.add_argument("input", help="音视频文件路径")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help=f"模型目录")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="设备")
    parser.add_argument("--compute-type", default="float16", help="计算类型")
    parser.add_argument("--beam-size", type=int, default=5, help="beam_size")
    parser.add_argument("--out-dir", default=None, help="输出目录")
    parser.add_argument("--verbose", action="store_true", help="显示预览")
    parser.add_argument("--remove-filler", action="store_true", help="清理填充词")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ 找不到文件: {args.input}")
        sys.exit(1)

    try:
        transcribe_file(
            args.input,
            model_dir=args.model_dir,
            device=args.device,
            compute_type=args.compute_type,
            beam_size=args.beam_size,
            out_dir=args.out_dir,
            verbose=args.verbose,
            remove_filler=args.remove_filler,
        )
    except Exception:
        print("❌ 转写过程中出现异常：")
        traceback.print_exc()
        sys.exit(1)

    print("🎉 任务结束。")


if __name__ == "__main__":
    main()
