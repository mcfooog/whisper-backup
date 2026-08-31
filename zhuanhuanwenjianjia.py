#  python zhuanhuanwenjianjia.py "D:\yt"
#   python zhuanhuanwenjianjia.py "E:\下载\YK\2025及以前"

import os
import sys
import subprocess
import traceback

# 支持的媒体格式（视频 + 音频）
MEDIA_EXTENSIONS = {
    # —— 视频 ——
    ".mp4",
    ".m4v",
    ".mkv",
    ".mov",
    ".avi",
    ".flv",
    ".f4v",
    ".wmv",
    ".asf",
    ".ts",
    ".m2ts",
    ".m2t",
    ".mts",
    ".tp",
    ".trp",
    ".tts",
    ".webm",
    ".ogv",
    ".rm",
    ".rmvb",
    ".ra",
    ".ram",
    ".3gp",
    ".3g2",
    ".3gpp",
    ".3gp2",
    ".mpg",
    ".mpeg",
    ".mp2",
    ".mpe",
    ".mpv",
    ".m1v",
    ".m2v",
    ".vob",
    ".dat",
    ".divx",
    ".xvid",
    ".mts2",
    ".m4p",
    ".m4b",
    ".m4r",
    ".qt",
    ".amv",
    ".nsv",
    ".drc",
    ".gifv",
    ".yuv",
    ".vivo",
    ".bik",
    ".roq",
    ".smk",
    # —— 音频 ——
    ".mp3",
    ".mp2",
    ".mp1",
    ".mpa",
    ".wav",
    ".wave",
    ".m4a",
    ".m4b",
    ".m4p",
    ".m4r",
    ".aac",
    ".ac3",
    ".aif",
    ".aifc",
    ".aiff",
    ".au",
    ".snd",
    ".flac",
    ".alac",
    ".ape",
    ".mac",
    ".wv",
    ".wvc",
    ".ogg",
    ".oga",
    ".opus",
    ".speex",
    ".wma",
    ".wax",
    ".amr",
    ".awb",
    ".gsm",
    ".dss",
    ".dct",
    ".dff",
    ".dsf",
    ".dsd",
    ".tta",
    ".tak",
    ".mka",
    ".cda",
    ".mid",
    ".midi",
    ".rmi",
    ".kar",
    ".xmi",
    ".ape",
    ".ofr",
    ".ofs",
    ".pac",
    ".vqf",
    ".vql",
    ".vqe",
    ".aifr",
}


def find_media_files(folder_path):
    """只查找当前文件夹里的媒体文件，不递归子文件夹"""

    media_files = []

    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)

        # 跳过子文件夹
        if not os.path.isfile(full_path):
            continue

        ext = os.path.splitext(file)[1].lower()

        if ext in MEDIA_EXTENSIONS:
            media_files.append(full_path)

    return sorted(media_files)


def main():
    if len(sys.argv) < 2:
        print("用法：")
        print('python zhuanhuanwenjianjia.py "D:\\yt"')
        sys.exit(1)

    folder_path = sys.argv[1]

    if not os.path.exists(folder_path):
        print(f"❌ 文件夹不存在: {folder_path}")
        sys.exit(1)

    if not os.path.isdir(folder_path):
        print(f"❌ 这不是文件夹: {folder_path}")
        sys.exit(1)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    zhuanwenzi_path = os.path.join(current_dir, "zhuanwenzi.py")

    if not os.path.exists(zhuanwenzi_path):
        print(f"❌ 找不到 zhuanwenzi.py : {zhuanwenzi_path}")
        sys.exit(1)

    print(f"📂 正在扫描文件夹: {folder_path}")

    media_files = find_media_files(folder_path)

    if not media_files:
        print("❌ 没找到媒体文件")
        sys.exit(1)

    print(f"🎬 共找到 {len(media_files)} 个媒体文件")
    print()

    success_count = 0
    fail_count = 0

    for index, media_file in enumerate(media_files, start=1):
        print("=" * 80)
        print(f"🎯 [{index}/{len(media_files)}]")
        print(f"📹 文件: {media_file}")
        print("=" * 80)

        try:
            cmd = [
                sys.executable,
                zhuanwenzi_path,
                media_file
            ]

            result = subprocess.run(cmd)

            if result.returncode == 0:
                success_count += 1
                print("✅ 转写成功")
            else:
                fail_count += 1
                print("❌ 转写失败")

        except Exception:
            fail_count += 1
            print("❌ 运行异常：")
            traceback.print_exc()

        print()

    print("=" * 80)
    print("🎉 全部任务完成")
    print(f"✅ 成功: {success_count}")
    print(f"❌ 失败: {fail_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()