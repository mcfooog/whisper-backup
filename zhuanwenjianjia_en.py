
# python zhuanwenjianjia_en.py "D:\yt"


import os
import sys
import subprocess
import traceback

# 支持的视频格式
VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".flv",
    ".wmv",
    ".m4v",
    ".ts",
    ".webm"
}


def find_video_files(folder_path):
    """只查找当前文件夹里的视频文件，不递归子文件夹"""
    video_files = []
    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)

        # 跳过子文件夹
        if not os.path.isfile(full_path):
            continue

        ext = os.path.splitext(file)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            video_files.append(full_path)

    return sorted(video_files)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print('python zhuanhuan_folder_en.py "D:\\yt"')
        sys.exit(1)

    folder_path = sys.argv[1]

    if not os.path.exists(folder_path):
        print(f"❌ Folder does not exist: {folder_path}")
        sys.exit(1)

    if not os.path.isdir(folder_path):
        print(f"❌ Not a valid directory: {folder_path}")
        sys.exit(1)

    # 获取当前文件夹，并寻找英文转写脚本 zhuan_en.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    zhuanwenzi_path = os.path.join(current_dir, "zhuan_en.py")

    if not os.path.exists(zhuanwenzi_path):
        print(f"❌ Cannot find english transcription script: {zhuanwenzi_path}")
        print("💡 请确保先前修改的英文脚本命名为 'zhuan_en.py' 并放在同一个目录下。")
        sys.exit(1)

    print(f"📂 Scanning folder: {folder_path}")

    video_files = find_video_files(folder_path)

    if not video_files:
        print("❌ No video files found.")
        sys.exit(1)

    print(f"🎬 Found {len(video_files)} video file(s)")
    print()

    success_count = 0
    fail_count = 0

    for index, video_file in enumerate(video_files, start=1):
        print("=" * 80)
        print(f"🎯 [{index}/{len(video_files)}]")
        print(f"📹 Processing: {video_file}")
        print("=" * 80)

        try:
            # 组装命令行，默认开启了 --remove-filler 剔除英文口头禅
            # 如果不想要剔除口头禅，可以把 "--remove-filler" 删掉
            cmd = [
                sys.executable,
                zhuanwenzi_path,
                video_file,
                "--remove-filler"
            ]

            # 运行英文转写脚本
            result = subprocess.run(cmd)

            if result.returncode == 0:
                success_count += 1
                print("✅ Transcription Success (转写成功)")
            else:
                fail_count += 1
                print("❌ Transcription Failed (转写失败)")

        except Exception:
            fail_count += 1
            print("❌ Execution Exception (运行异常)：")
            traceback.print_exc()

        print()

    print("=" * 80)
    print("🎉 All Tasks Completed")
    print(f"✅ Success: {success_count}")
    print(f"❌ Failed: {fail_count}")
    print("=" * 80)


if __name__ == "__main__":
    main()