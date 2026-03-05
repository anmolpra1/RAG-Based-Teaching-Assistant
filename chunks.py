import whisper
import torch
import json
import os

# Select GPU if available, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load small Whisper model
model = whisper.load_model("small", device=device)

audio_dir = "audios"
output_dir = "jsons"

# Create output folder if missing
os.makedirs(output_dir, exist_ok=True)

for audio in os.listdir(audio_dir):

    # Process only valid audio files
    if not audio.lower().endswith((".mp3", ".wav", ".m4a", ".flac")):
        continue

    # Expect filename format: number_title.ext
    if "_" not in audio:
        continue

    number, rest = audio.split("_", 1)
    title, _ = os.path.splitext(rest)

    try:
        # Transcribe Hindi → English
        result = model.transcribe(
            audio=os.path.join(audio_dir, audio),
            language="hi",
            task="translate"
        )

        # Collect segment data
        chunks = [
            {
                "number": number,
                "title": title,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip()
            }
            for seg in result["segments"]
        ]

        output_data = {
            "chunks": chunks,
            "full_text": result["text"].strip()
        }

        # Save JSON
        output_path = os.path.join(
            output_dir,
            os.path.splitext(audio)[0] + ".json"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

        print(f"Done: {audio}")

    except Exception as e:
        print(f"Error: {audio} -> {e}")

