import os
import re
import uuid
import json
import tempfile
import subprocess
import shutil
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import groq

# ─────────────────────────── app setup ──────────────────────────

app = FastAPI(title="SubtitleForge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

GROQ_WHISPER_MODEL = "whisper-large-v3"
GROQ_LLM_MODEL = "qwen-qwq-32b"

UPLOAD_DIR = Path(tempfile.gettempdir()) / "subtitle_forge"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGE_CODE_MAP = {
    "english": "en",
    "tamil": "ta",
    "telugu": "te",
}


# ─────────────────────────── ffmpeg helpers ──────────────────────

def extract_audio(video_path: Path, audio_path: Path) -> None:
    """Extract mono 16 kHz WAV — optimal for Whisper."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")


def get_audio_duration(audio_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def split_audio(audio_path: Path, chunk_dir: Path, chunk_seconds: int = 600) -> list[Path]:
    """Split audio into fixed-length WAV chunks."""
    chunk_dir.mkdir(exist_ok=True)
    pattern = str(chunk_dir / "chunk_%04d.wav")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(audio_path),
        "-f", "segment",
        "-segment_time", str(chunk_seconds),
        "-c", "copy",
        pattern,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return sorted(chunk_dir.glob("chunk_*.wav"))


# ─────────────────────────── transcription ──────────────────────

def _groq_transcribe_single(client: groq.Groq, audio_path: Path, lang: str) -> list[dict]:
    """Transcribe one WAV file via Groq Whisper. File must be < 25 MB."""
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            file=(audio_path.name, f, "audio/wav"),
            model=GROQ_WHISPER_MODEL,
            language=lang,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    return [
        {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
        }
        for seg in response.segments
    ]


def transcribe_with_groq(audio_path: Path, lang: str) -> list[dict]:
    """
    Transcribe audio with Groq Whisper.
    Auto-chunks if file > 24 MB to stay within Groq's 25 MB limit.
    """
    client = groq.Groq(api_key=GROQ_API_KEY)
    size_mb = audio_path.stat().st_size / (1024 * 1024)

    if size_mb <= 24:
        return _groq_transcribe_single(client, audio_path, lang)

    chunk_dir = audio_path.parent / "chunks"
    chunks = split_audio(audio_path, chunk_dir)
    all_segments: list[dict] = []
    time_offset = 0.0

    for chunk in chunks:
        segs = _groq_transcribe_single(client, chunk, lang)
        for seg in segs:
            all_segments.append({
                "start": round(seg["start"] + time_offset, 3),
                "end": round(seg["end"] + time_offset, 3),
                "text": seg["text"],
            })
        time_offset += get_audio_duration(chunk)

    shutil.rmtree(chunk_dir, ignore_errors=True)
    return all_segments


def transcribe_with_local_whisper(audio_path: Path, lang: str) -> list[dict]:
    """
    Fallback: openai-whisper (pure Python, no C libs needed).
    Only runs if Groq is down or rate-limited.
    """
    try:
        import whisper
        model = whisper.load_model("small")
        result = model.transcribe(str(audio_path), language=lang, verbose=False)
        return [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip(),
            }
            for seg in result["segments"]
        ]
    except ImportError:
        raise RuntimeError(
            "Groq is unavailable and no local fallback is installed. "
            "Please check your GROQ_API_KEY on Render."
        )
    except Exception as exc:
        raise RuntimeError(f"Local whisper fallback failed: {exc}") from exc


def transcribe(audio_path: Path, lang: str) -> list[dict]:
    """Try Groq first; fall back to local whisper."""
    if not GROQ_API_KEY:
        print("[INFO] No GROQ_API_KEY — using local whisper directly")
        return transcribe_with_local_whisper(audio_path, lang)
    try:
        return transcribe_with_groq(audio_path, lang)
    except Exception as exc:
        print(f"[WARN] Groq failed ({exc}), falling back to local whisper")
        return transcribe_with_local_whisper(audio_path, lang)


# ─────────────────────────── translation ────────────────────────

def translate_segments(segments: list[dict], source_lang: str) -> list[dict]:
    """
    Translate segment texts → English using Groq + Qwen-QwQ-32b.
    Processes in batches of 30 to stay within token limits.
    """
    client = groq.Groq(api_key=GROQ_API_KEY)
    BATCH_SIZE = 30
    translated: list[dict] = []

    for i in range(0, len(segments), BATCH_SIZE):
        batch = segments[i: i + BATCH_SIZE]
        numbered = "\n".join(
            f"{j + 1}. {seg['text']}" for j, seg in enumerate(batch)
        )
        prompt = (
            f"You are a professional subtitle translator. "
            f"Translate the following {source_lang} subtitle lines into English. "
            f"Output ONLY the translated lines, numbered exactly as given. "
            f"Do not include reasoning, explanations, or <think> blocks. "
            f"Output format: '1. translated text'\n\n"
            f"{numbered}"
        )
        response = client.chat.completions.create(
            model=GROQ_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()

        # Strip Qwen's <think>...</think> reasoning blocks
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # Parse "N. text" lines
        translated_texts: list[str] = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            if ". " in line and line.split(". ")[0].isdigit():
                translated_texts.append(line.split(". ", 1)[1])
            else:
                translated_texts.append(line)

        # Pad/trim to exact batch length to stay in sync with timestamps
        while len(translated_texts) < len(batch):
            translated_texts.append(batch[len(translated_texts)]["text"])
        translated_texts = translated_texts[: len(batch)]

        for seg, text in zip(batch, translated_texts):
            translated.append({**seg, "text": text})

    return translated


# ─────────────────────────── SRT helpers ────────────────────────

def _ts(seconds: float) -> str:
    """Float seconds → SRT timestamp HH:MM:SS,mmm"""
    ms = int(round((seconds % 1) * 1000))
    s  = int(seconds) % 60
    m  = (int(seconds) // 60) % 60
    h  = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments: list[dict]) -> str:
    blocks = []
    for i, seg in enumerate(segments, 1):
        blocks.append(f"{i}\n{_ts(seg['start'])} --> {_ts(seg['end'])}\n{seg['text']}")
    return "\n\n".join(blocks) + "\n"


# ─────────────────────────── routes ─────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "groq_configured": bool(GROQ_API_KEY)}


@app.post("/upload/init")
async def upload_init(body: dict):
    """Create a job directory and return a job_id."""
    job_id  = uuid.uuid4().hex
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True)
    meta = {
        "filename": body.get("filename", "input.mp4"),
        "total_chunks": body["total_chunks"],
    }
    (job_dir / "meta.json").write_text(json.dumps(meta))
    return {"job_id": job_id}


@app.post("/upload/chunk")
async def upload_chunk(
    job_id: str = Form(...),
    chunk_index: int = Form(...),
    chunk: UploadFile = File(...),
):
    """Receive one chunk and save it to the job directory."""
    job_dir = UPLOAD_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404, "Job not found")

    chunk_path = job_dir / f"chunk_{chunk_index:05d}"
    with open(chunk_path, "wb") as f:
        while True:
            data = await chunk.read(1 * 1024 * 1024)
            if not data:
                break
            f.write(data)

    return {"received": chunk_index}


@app.post("/process")
async def process_job(body: dict):
    """Assemble chunks, extract audio, transcribe, translate, return SRT."""
    job_id   = body.get("job_id")
    src_lang = body.get("source_language", "english").lower().strip()
    tgt_lang = body.get("target_language", "english").lower().strip()
    filename = body.get("filename", "output.mp4")

    if not job_id:
        raise HTTPException(400, "job_id is required")

    job_dir = UPLOAD_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(404, "Job not found")

    if src_lang not in LANGUAGE_CODE_MAP:
        raise HTTPException(400, f"Unsupported language: {src_lang}")

    lang_code  = LANGUAGE_CODE_MAP[src_lang]
    suffix     = Path(filename).suffix or ".mp4"
    video_path = job_dir / f"input{suffix}"
    audio_path = job_dir / "audio.wav"
    srt_path   = job_dir / "output.srt"

    try:
        # 1. Assemble chunks in order
        chunk_files = sorted(job_dir.glob("chunk_*"))
        if not chunk_files:
            raise HTTPException(422, "No chunks received")

        with open(video_path, "wb") as out:
            for cf in chunk_files:
                out.write(cf.read_bytes())
                cf.unlink()

        # 2. Extract audio
        extract_audio(video_path, audio_path)
        video_path.unlink(missing_ok=True)

        # 3. Transcribe
        segments = transcribe(audio_path, lang_code)
        audio_path.unlink(missing_ok=True)

        if not segments:
            raise HTTPException(422, "No speech detected in the file.")

        # 4. Translate if needed
        if src_lang != "english" and tgt_lang == "english":
            segments = translate_segments(segments, src_lang)

        # 5. Write SRT
        srt_path.write_text(segments_to_srt(segments), encoding="utf-8")

        stem = Path(filename).stem
        return FileResponse(
            path=str(srt_path),
            media_type="text/plain",
            filename=f"{stem}.srt",
            headers={"X-Job-Id": job_id},
        )

    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(500, str(exc)) from exc
