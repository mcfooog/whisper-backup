import os
import sys
import re
import time
import socket


def parse_srt(content):
    subs = []
    blocks = re.split(r'\n\n', content.strip())
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            try:
                index = int(lines[0].strip())
                time_line = lines[1].strip()
                text = '\n'.join(lines[2:]).strip()
                translation = ''
                if len(lines) >= 4:
                    translation = lines[2].strip()
                    text = '\n'.join(lines[3:]).strip()
                subs.append({
                    'index': index,
                    'time': time_line,
                    'text': text,
                    'translation': translation
                })
            except ValueError:
                continue
    return subs


def format_srt(subs):
    lines = []
    for sub in subs:
        lines.append(str(sub['index']))
        lines.append(sub['time'])
        if sub.get('translation'):
            lines.append(sub['translation'])
        lines.append(sub['text'])
        lines.append('')
    return '\n'.join(lines).strip()


def translate_text(text):
    if not text.strip():
        return ""
    
    socket.setdefaulttimeout(15)
    
    for attempt in range(3):
        try:
            from deep_translator import GoogleTranslator
            translator = GoogleTranslator(source='ja', target='zh-CN')
            result = translator.translate(text)
            return result
        except Exception as e:
            time.sleep(3)
            if attempt == 2:
                print(f"⚠️ 翻译失败 '{text[:30]}...'")
                return ""


def main():
    if len(sys.argv) < 2:
        print("用法：")
        print('python translate_srt.py "input.srt" [batch_size]')
        print('示例：python translate_srt.py "video_jp.srt" 100')
        sys.exit(1)

    input_path = sys.argv[1]
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    if not os.path.exists(input_path):
        print(f"❌ 找不到文件: {input_path}")
        sys.exit(1)

    dirname = os.path.dirname(input_path)
    basename = os.path.basename(input_path)
    name, ext = os.path.splitext(basename)
    output_filename = f"{name}_cn{ext}"
    output_path = os.path.join(dirname, output_filename)

    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()
        subs = parse_srt(content)
        print(f"📖 已存在翻译文件，共 {len(subs)} 条字幕")
    else:
        with open(input_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()
        subs = parse_srt(content)
        print(f"📖 共解析到 {len(subs)} 条字幕")

    already_done = sum(1 for s in subs if s.get('translation'))
    remaining = [i for i, s in enumerate(subs) if not s.get('translation')]
    
    print(f"✅ 已翻译: {already_done} 条")
    print(f"🔄 待翻译: {len(remaining)} 条")
    print(f"📦 本次处理: {batch_size} 条")

    if not remaining:
        print("\n🎉 所有字幕已翻译完成！")
        print(f"📄 最终文件: {output_path}")
        return

    batch = remaining[:batch_size]
    start_idx = batch[0] + 1
    end_idx = batch[-1] + 1
    
    print(f"\n📋 开始处理第 {start_idx} - {end_idx} 条字幕...")

    success_count = 0
    fail_count = 0

    for idx in batch:
        sub = subs[idx]
        translation = translate_text(sub['text'])
        if translation:
            sub['translation'] = translation
            success_count += 1
        else:
            fail_count += 1

    with open(output_path, 'w', encoding='utf-8-sig') as f:
        f.write(format_srt(subs))

    total_done = already_done + success_count
    remaining_after = len(subs) - total_done
    
    print(f"\n✅ 本批完成: 成功 {success_count} 条, 失败 {fail_count} 条")
    print(f"📊 总计完成: {total_done}/{len(subs)} ({total_done/len(subs)*100:.1f}%)")
    print(f"💾 已保存到: {output_path}")
    
    if remaining_after > 0:
        print(f"\n📝 还剩 {remaining_after} 条待翻译")
        print(f"🔄 下次运行相同命令将从第 {end_idx + 1} 条继续")
    else:
        print("\n🎉 所有字幕已翻译完成！")


if __name__ == "__main__":
    main()
