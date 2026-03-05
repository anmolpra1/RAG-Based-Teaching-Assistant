import os
import whisper

AUDIO_DIR = "audios"
TEXT_DIR = "transcripts"

# Create output folder if it doesn't exist
os.makedirs(TEXT_DIR, exist_ok=True)

# Load model (practical choice)
model = whisper.load_model("small")  

# Iterate over all MP3 files in the audio directory
for file in os.listdir(AUDIO_DIR):

    if not file.lower().endswith(".mp3"):
        continue

    audio_path = os.path.join(AUDIO_DIR, file)
    output_path = os.path.join(
        TEXT_DIR,
        os.path.splitext(file)[0] + ".txt"
    )

    # Skip if already translated
    if os.path.exists(output_path):
        print(f"Skipping (already done): {file}")
        continue

    print(f"Translating: {file}")

    result = model.transcribe(
        audio_path,
        task="translate"   # speech → English
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result["text"])

print("✅ All MP3 files translated")
