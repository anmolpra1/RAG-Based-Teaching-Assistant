import os
import subprocess

VIDEO_DIR = "videos"
AUDIO_DIR = "audios"

# Ensure output directory exists
os.makedirs(AUDIO_DIR, exist_ok=True)

files = os.listdir(VIDEO_DIR)

for file in files:
    # Skip non-video files
    if not file.lower().endswith((".mp4", ".mkv", ".avi", ".mov")):
        continue

    print("Processing:", file)

    # Safe filename without extension
    base_name = os.path.splitext(file)[0]

    input_path = os.path.join(VIDEO_DIR, file)
    output_path = os.path.join(AUDIO_DIR, base_name + ".mp3")

    subprocess.run([
        "ffmpeg",
        "-y",          # overwrite if file exists
        "-i", input_path,
        "-vn",         # no video
        "-ab", "192k",
        output_path
    ])

print("✅ All videos converted")
