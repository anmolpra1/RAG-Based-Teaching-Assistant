# 🎓 RAG-Based AI Teaching Assistant

A local, fully offline RAG (Retrieval-Augmented Generation) pipeline that turns Hindi lecture videos into a searchable, conversational AI assistant. Ask questions about the course and get timestamped answers pointing you to the exact video and moment.

---

## 🧠 How It Works

```
Videos (.mp4)
    ↓  process_videos.py   — extract audio via ffmpeg
Audio (.mp3)
    ↓  chunks.py           — transcribe & translate (Hindi → English) using Whisper
JSON chunks (.json)
    ↓  read_chunks.py      — embed each chunk using bge-m3 via Ollama
embeddings.parquet
    ↓  search.py           — cosine similarity search + LLaMA 3.2 inference
Answer with video + timestamp
```

---

## 📁 Project Structure

```
.
├── videos/                  # Raw lecture videos (.mp4, .mkv, etc.)
├── audios/                  # Extracted audio files (.mp3)
├── jsons/                   # Whisper transcription chunks (.json)
├── transcripts/             # Plain text transcriptions (.txt)
├── embeddings.parquet       # Precomputed chunk embeddings
├── process_videos.py        # Step 1: Video → Audio (ffmpeg)
├── chunks.py                # Step 2: Audio → JSON chunks (Whisper)
├── speech_to_text.py        # Optional: Audio → plain text transcripts
├── read_chunks.py           # Step 3: JSON chunks → embeddings (bge-m3)
├── search.py                # Step 4: Query → RAG answer (LLaMA 3.2)
├── prompt.txt               # Last prompt sent to LLM (debug)
└── response.txt             # Last LLM response (debug)
```

---

## ⚙️ Prerequisites

### System
- Python 3.9+
- [ffmpeg](https://ffmpeg.org/download.html) — must be on your `PATH`
- [Ollama](https://ollama.com/) — local LLM & embedding server

### Ollama Models
Pull both models before running:
```bash
ollama pull bge-m3          # Embedding model
ollama pull llama3.2:3b     # Chat/inference model
```

Start the Ollama server:
```bash
ollama serve
```

---

## 🚀 Setup & Usage

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Extract Audio from Videos
Place your `.mp4` / `.mkv` / `.avi` / `.mov` files in the `videos/` folder, then run:
```bash
python process_videos.py
```
Outputs `.mp3` files to `audios/`.

### 3. Transcribe & Chunk Audio
```bash
python chunks.py
```
Uses Whisper (`small` model) to transcribe Hindi audio and translate to English. Saves timestamped JSON chunks to `jsons/`.

> **Note:** Audio files must follow the naming format: `<number>_<title>.mp3`  
> Example: `01_intro_to_html.mp3`

### 4. Generate Embeddings
```bash
python read_chunks.py
```
Embeds each chunk using `bge-m3` via Ollama and saves the result to `embeddings.parquet`.

### 5. Ask Questions
```bash
python search.py
```
Starts an interactive Q&A loop. Type your question and get an answer with the relevant video title and timestamp.

```
Ask a Question (or 'quit' to exit): What is the box model in CSS?

--- Answer ---
The box model is covered in Video 12 "CSS Basics" around 4:32. It describes how every HTML element...
```

---

## 🔍 Optional: Plain Text Transcripts

To generate simple `.txt` transcripts (no chunking) from audio files:
```bash
python speech_to_text.py
```
Outputs to `transcripts/`.

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Video → Audio | ffmpeg |
| Speech-to-text + Translation | OpenAI Whisper (`small`) |
| Embeddings | `bge-m3` via Ollama |
| Vector similarity | scikit-learn cosine similarity |
| LLM Inference | LLaMA 3.2 (3B) via Ollama |
| Data storage | Pandas + Parquet |

---

## 📝 Notes

- All processing runs **locally** — no API keys or internet required after setup.
- The assistant is scoped to answer only course-related questions.
- GPU is used automatically if CUDA is available (for Whisper).
- `prompt.txt` and `response.txt` are written after each query for debugging.
