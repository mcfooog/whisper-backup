# python zhuan_paraform.py
# 基于 FunASR Paraformer 的纯文本转写脚本（防重复增强版 + 标点恢复 + 长音频显存优化 + 输出 Markdown）

import os
import sys
import re
import argparse
import traceback

# ================= 显存优化环境变量 =================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# ===================================================

# 默认模型路径（指向当前目录）
DEFAULT_MODEL_DIR = r"C:\whisper\paraformer-zh"
# 默认标点模型路径
DEFAULT_PUNC_MODEL_DIR = r"C:\whisper\ct-punc\models\iic--punc_ct-transformer_cn-en-common-vocab471067-large\snapshots\master"


def get_available_filename(base_path: str) -> str:
    """如果文件存在，自动生成不重复文件名（name(1).md）"""
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


def try_import_s2t_tools():
    """尝试导入用于繁转简的库"""
    try:
        from opencc import OpenCC
        cc = OpenCC("t2s")
        return cc.convert, True
    except Exception:
        pass
    try:
        import zhconv
        return lambda s: zhconv.convert(s, "zh-cn"), True
    except Exception:
        pass
    try:
        from hanziconv import HanziConv
        return lambda s: HanziConv.toSimplified(s), True
    except Exception:
        pass
    return lambda s: s, False


def clean_common_filler(text: str) -> str:
    """清理常见填充词"""
    text = re.sub(r'^(?:\s*(?:uh|um|ah|eh|erm|ahh|uhh|…|\.|\s)+\s*){2,}$',
                  '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'(\b(?:uh|um|ah|eh|erm)\b)(?:[\s.…·，,]*){2,}',
                  ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def split_sentences(text: str) -> list:
    """将文本按句子分割（中英文标点）"""
    if not text.strip():
        return []
    delimiter_pattern = r'(?<=[。！？；!?;])'
    sentences = re.split(delimiter_pattern, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) <= 1 and ' ' in text:
        sentences = [seg.strip() for seg in text.split(' ') if seg.strip()]
    return sentences


def normalize_sentence(s: str) -> str:
    """标准化句子：去除首尾空格和句尾标点，用于去重比较"""
    s = s.strip()
    s = re.sub(r'[。！？!?]+$', '', s)
    return s


def remove_consecutive_duplicates(sentences: list) -> list:
    """去除连续重复的句子（保留第一句）"""
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
    punc_model_dir: str = None,
    device: str = "cuda",
    batch_size_s: int = 60,
    out_dir: str = None,
    verbose: bool = False,
    convert_to_simplified: bool = True,
    remove_filler: bool = False,
):
    try:
        from funasr import AutoModel
    except Exception as e:
        raise RuntimeError(
            "无法导入 funasr。请确认已安装：pip install funasr modelscope torch"
        ) from e

    print(f"🚀 正在加载 ASR 模型: {model_dir} (device={device}) ...", flush=True)
    if punc_model_dir:
        print(f"🚀 正在加载标点模型: {punc_model_dir} ...", flush=True)

    try:
        model = AutoModel(
            model=model_dir,
            punc_model=punc_model_dir,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=device,
            disable_update=True
        )
    except Exception as e:
        print("❌ 模型加载失败（请确认模型路径、CUDA 驱动与显存是否正常）。")
        raise

    print(f"🎙️ 开始转写: {input_path}", flush=True)
    print(f"⏳ Paraformer 正在推理中 (batch_size_s={batch_size_s})，请稍候...", flush=True)

    try:
        res = model.generate(
            input=input_path,
            batch_size_s=batch_size_s,
        )
        full_text = res[0]["text"].strip()
    except Exception as e:
        print("❌ 推理过程中出现异常：")
        traceback.print_exc()
        sys.exit(1)

    if not full_text:
        print("⚠️ 识别结果为空，请检查音频文件是否包含有效语音。", flush=True)
        return None

    if remove_filler:
        full_text = clean_common_filler(full_text)

    convert_fn, conv_available = try_import_s2t_tools()
    if convert_to_simplified:
        if conv_available:
            try:
                full_text = convert_fn(full_text)
            except Exception:
                print("⚠️ 繁转简转换时出错，已跳过转换。", flush=True)
        else:
            print("⚠️ 未检测到转换库，未进行繁简转换。", flush=True)

    sentences = split_sentences(full_text)
    original_len = len(sentences)
    sentences = remove_consecutive_duplicates(sentences)
    removed = original_len - len(sentences)
    if removed > 0:
        print(f"🧹 后处理去除了 {removed} 句连续重复的内容", flush=True)

    # 构建 Markdown 内容：一级标题 + 正文
    markdown_lines = ["# 转写文本\n"]
    markdown_lines.extend(sentences)
    output_text = "\n".join(markdown_lines)
    output_text = re.sub(r'\n\s*\n+', '\n', output_text).strip()
    if output_text and not output_text.endswith('\n'):
        output_text += '\n'

    # 输出文件后缀改为 .md
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(input_path))[0] + ".md"
        out_file_candidate = os.path.join(out_dir, base_name)
    else:
        out_file_candidate = os.path.splitext(input_path)[0] + ".md"

    out_file = get_available_filename(out_file_candidate)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(f"✅ 转写完成，已保存: {os.path.abspath(out_file)}", flush=True)
    return os.path.abspath(out_file)


def main():
    parser = argparse.ArgumentParser(description="基于 FunASR Paraformer 的纯文本转写（防重复增强版 + 标点 + 显存优化 + Markdown输出）")
    parser.add_argument("input", help="音视频文件路径")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help=f"ASR 模型目录")
    parser.add_argument("--punc-model-dir", default=DEFAULT_PUNC_MODEL_DIR, help=f"标点模型目录")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="设备")
    parser.add_argument("--batch-size-s", type=int, default=60, help="批处理音频总时长(秒)，默认60，显存不足可调低至30")
    parser.add_argument("--out-dir", default=None, help="输出目录")
    parser.add_argument("--verbose", action="store_true", help="显示预览")
    parser.add_argument("--no-s2t", action="store_true", help="关闭繁转简")
    parser.add_argument("--remove-filler", action="store_true", help="清理填充词")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ 找不到文件: {args.input}")
        sys.exit(1)

    try:
        transcribe_file(
            args.input,
            model_dir=args.model_dir,
            punc_model_dir=args.punc_model_dir,
            device=args.device,
            batch_size_s=args.batch_size_s,
            out_dir=args.out_dir,
            verbose=args.verbose,
            convert_to_simplified=not args.no_s2t,
            remove_filler=args.remove_filler,
        )
    except Exception:
        print("❌ 转写过程中出现异常：")
        traceback.print_exc()
        sys.exit(1)

    print("🎉 任务结束。")


if __name__ == "__main__":
    main()