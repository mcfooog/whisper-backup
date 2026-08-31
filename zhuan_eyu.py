# python zhuan_eyu.py "D:\yt\video.mp4"

import os
import sys
import re
import argparse
import traceback
import socket
import time
import json

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


def split_long_sentence(sentence: str, max_length: int = 150) -> list:
    if len(sentence) <= max_length:
        return [sentence]
    parts = []
    current = ""
    for word in sentence.split():
        if len(current) + len(word) + 1 <= max_length:
            current += word + " "
        else:
            parts.append(current.strip())
            current = word + " "
    if current.strip():
        parts.append(current.strip())
    return parts


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


def is_russian_text(text):
    russian_chars = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ'
    return any(char in russian_chars for char in text)


def translate_text(text, translator=None):
    if not text.strip():
        return ""

    socket.setdefaulttimeout(30)

    if translator is None:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='ru', target='zh-CN')

    for attempt in range(5):
        try:
            result = translator.translate(text)
            if result and not is_russian_text(result):
                return result
            time.sleep(5)
        except Exception as e:
            time.sleep(5)
            if attempt == 4:
                print(f"⚠️ 翻译失败 '{text[:50]}...'")
                return text
    return text


def transcribe_file(
    input_path: str,
    model_dir: str,
    device: str = "cuda",
    compute_type: str = "float16",
    beam_size: int = 5,
    out_dir: str = None,
    verbose: bool = False,
    remove_filler: bool = False,
    translate: bool = True,
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

    print(f"🎙️ 开始转写俄语: {input_path}", flush=True)

    transcribe_kwargs = {
        "beam_size": beam_size,
        "language": "ru",
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
            language="ru",
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

    short_sentences = []
    for s in sentences:
        short_sentences.extend(split_long_sentence(s, max_length=150))
    print(f"📝 句子分割: {len(sentences)} 句 → {len(short_sentences)} 短句", flush=True)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_path))[0] + "_eyu_cn.txt"
        out_file_candidate = os.path.join(out_dir, base_name)
    else:
        out_file_candidate = os.path.splitext(input_path)[0] + "_eyu_cn.txt"

    out_file = get_available_filename(out_file_candidate)
    cache_file = out_file.replace(".txt", "_cache.json")

    cached_translations = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_translations = json.load(f)
            print(f"📋 加载了 {len(cached_translations)} 条缓存翻译", flush=True)
        except Exception:
            cached_translations = {}

    if translate:
        print(f"🌍 开始翻译 {len(short_sentences)} 句俄语到中文...", flush=True)

        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='ru', target='zh-CN')

        translated_sentences = []
        success_count = 0
        fail_count = 0

        for i, sentence in enumerate(short_sentences, 1):
            sentence_key = sentence.strip()

            if sentence_key in cached_translations:
                translation = cached_translations[sentence_key]
            else:
                translation = translate_text(sentence, translator)
                cached_translations[sentence_key] = translation
                time.sleep(0.5)

            if translation and not is_russian_text(translation):
                translated_sentences.append(translation)
                success_count += 1
            else:
                translated_sentences.append(translation)
                fail_count += 1

            if i % 10 == 0:
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cached_translations, f, ensure_ascii=False, indent=2)
                print(f"📊 翻译进度: {i}/{len(short_sentences)} ({i/len(short_sentences)*100:.1f}%), 成功: {success_count}, 失败: {fail_count}", flush=True)

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cached_translations, f, ensure_ascii=False, indent=2)

        output_text = "\n".join(translated_sentences)
        print(f"✅ 翻译完成: 成功 {success_count} 条, 失败 {fail_count} 条", flush=True)
    else:
        output_text = "\n".join(short_sentences)

    output_text = re.sub(r'\n\s*\n+', '\n', output_text).strip()
    if output_text and not output_text.endswith('\n'):
        output_text += '\n'

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(output_text)

    try:
        lang = getattr(info, "language", None)
        prob = getattr(info, "language_probability", None)
        if lang:
            print(f"🌍 检测语言: {lang} (概率 {prob:.2f})", flush=True)
    except Exception:
        pass

    print(f"✅ 俄语转写并翻译完成，已保存: {os.path.abspath(out_file)}", flush=True)
    return os.path.abspath(out_file)


def main():
    parser = argparse.ArgumentParser(description="基于 faster-whisper 的俄语转写并翻译为中文")
    parser.add_argument("input", help="音视频文件路径")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help=f"模型目录")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="设备")
    parser.add_argument("--compute-type", default="float16", help="计算类型")
    parser.add_argument("--beam-size", type=int, default=5, help="beam_size")
    parser.add_argument("--out-dir", default=None, help="输出目录")
    parser.add_argument("--verbose", action="store_true", help="显示预览")
    parser.add_argument("--remove-filler", action="store_true", help="清理填充词")
    parser.add_argument("--no-translate", action="store_true", help="只转写不翻译")
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
            translate=not args.no_translate,
        )
    except Exception:
        print("❌ 转写过程中出现异常：")
        traceback.print_exc()
        sys.exit(1)

    print("🎉 任务结束。")


if __name__ == "__main__":
    main()
