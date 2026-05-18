from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, HTMLResponse, StreamingResponse, FileResponse
from starlette.background import BackgroundTask
import asyncio
from typing import Optional
import base64
import secrets
import subprocess
import tempfile
import os
import json
import sys
import re
import html as _H
import difflib
import httpx
import threading
import time
import requests as _req
from pathlib import Path
import shutil
from starlette.middleware.base import BaseHTTPMiddleware

try:
    from gradio_client import Client as GradioClient, handle_file as gradio_handle_file
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

app = FastAPI()

# ── Job store for async generation ───────────────────────────────────────────
_jobs: dict[str, dict] = {}
_gen_sem = threading.Semaphore(1)   # one generation at a time across all video types (OOM prevention)
RESULTS_DIR = Path("/tmp/easymh_results")
RESULTS_DIR.mkdir(exist_ok=True)

def _job_log(job_id: str, msg: str):
    if job_id in _jobs:
        _jobs[job_id]["logs"].append(msg)
        print(f"[job:{job_id}] {msg}", flush=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VIDEO_WIDTH       = 720
VIDEO_HEIGHT      = 1280
MAX_SUBTITLE_CHARS = 65
OPENAI_MODEL      = "gpt-4o"

FONT_DIR   = Path("/usr/share/fonts/truetype/montserrat")
FRAMES_DIR = Path("/app/frames")
MUSIC_DIR  = Path("/app/music")
FONT_MAP = {
    "montserrat":          FONT_DIR / "Montserrat-Bold.ttf",
    "montserrat bold":     FONT_DIR / "Montserrat-Bold.ttf",
    "montserrat semibold": FONT_DIR / "Montserrat-SemiBold.ttf",
    "montserrat regular":  FONT_DIR / "Montserrat-Regular.ttf",
    "gilroy":              FONT_DIR / "Gilroy-Medium.ttf",
    "gilroy medium":       FONT_DIR / "Gilroy-Medium.ttf",
    "georgia":             FONT_DIR / "Georgia.ttf",
    "ltcarpet":            FONT_DIR / "LTCarpet.ttf",
    "lt carpet":           FONT_DIR / "LTCarpet.ttf",
    "bodyhand":            FONT_DIR / "Bodyhand Regular.otf",
    "inter":               FONT_DIR / "Inter-VariableFont_opsz,wght.ttf",
}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": str(exc.errors())})


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {str(exc)}"})


def hex_to_rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _fit_image(img, w: int, h: int):
    from PIL import Image
    scale = max(w / img.width, h / img.height)
    new_w, new_h = int(img.width * scale), int(img.height * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - w) // 2
    top  = (new_h - h) // 2
    return img.crop((left, top, left + w, top + h)).convert("RGB")


def load_font(font_name: str, size: int):
    from PIL import ImageFont
    key = font_name.lower().strip()
    path = FONT_MAP.get(key)
    if path and path.exists():
        return ImageFont.truetype(str(path), size)
    # fallback: search any .ttf/.otf that contains the first word of font name
    first_word = key.split()[0]
    for d in ["/usr/share/fonts", "/usr/local/share/fonts"]:
        for ext in ("*.ttf", "*.otf"):
            for f in Path(d).rglob(ext):
                if first_word in f.stem.lower():
                    return ImageFont.truetype(str(f), size)
    return ImageFont.load_default()


def wrap_lines(draw, text: str, font, max_width: int) -> list[str]:
    words = text.split()
    lines, cur = [], []
    for word in words:
        test = " ".join(cur + [word])
        if draw.textbbox((0, 0), test, font=font)[2] <= max_width:
            cur.append(word)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [word]
    if cur:
        lines.append(" ".join(cur))
    return lines


def fix_typography(lines: list[str]) -> list[str]:
    # Widow fix: last line = 1 short word → pull last word from prev line
    if len(lines) >= 2:
        last = lines[-1]
        if len(last.split()) == 1 and len(last.rstrip(".,!?")) <= 6:
            prev = lines[-2].split()
            if len(prev) > 1:
                lines[-2] = " ".join(prev[:-1])
                lines[-1] = prev[-1] + " " + last

    # Dash fix: line starts with dash → move dash to end of prev line
    fixed = []
    for i, line in enumerate(lines):
        if i > 0 and line.startswith(("—", "–", "-")):
            parts = line.split(None, 1)
            fixed[-1] += " " + parts[0]
            if len(parts) > 1:
                fixed.append(parts[1])
        else:
            fixed.append(line)
    return fixed


def split_long_blocks(transcript_data: list, font_name: str, font_size: int = 36) -> list:
    """Split blocks that still exceed MAX_LINES at the given font size."""
    from PIL import Image, ImageDraw
    MAX_LINES = 4
    max_width = int(VIDEO_WIDTH * 0.72)
    draw = ImageDraw.Draw(Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT)))
    font = load_font(font_name, font_size)

    result = []
    for block in transcript_data:
        text = block["text"]
        if len(wrap_lines(draw, text, font, max_width)) <= MAX_LINES:
            result.append(block)
            continue
        # Split word-by-word into chunks that fit MAX_LINES
        words = text.split()
        chunks, cur = [], []
        for word in words:
            test = " ".join(cur + [word])
            if len(wrap_lines(draw, test, font, max_width)) <= MAX_LINES:
                cur.append(word)
            else:
                if cur:
                    chunks.append(" ".join(cur))
                cur = [word]
        if cur:
            chunks.append(" ".join(cur))
        if len(chunks) <= 1:
            result.append(block)
            continue
        # Distribute time evenly across sub-blocks
        total_dur = block["end"] - block["start"]
        chunk_dur = total_dur / len(chunks)
        for i, chunk_text in enumerate(chunks):
            result.append({
                "text": chunk_text,
                "start": round(block["start"] + i * chunk_dur, 3),
                "end":   round(block["start"] + (i + 1) * chunk_dur, 3),
            })
    return result


def fit_font_size(texts: list, font_name: str, initial_size: int) -> int:
    """Find the largest font size where every text block fits within MAX_LINES."""
    from PIL import Image, ImageDraw
    draw = ImageDraw.Draw(Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT)))
    max_width = int(VIDEO_WIDTH * 0.72)
    MAX_LINES = 4
    MIN_SIZE  = 36
    size = initial_size
    for text in texts:
        if not text.strip():
            continue
        font = load_font(font_name, size)
        lines = wrap_lines(draw, text, font, max_width)
        while len(lines) > MAX_LINES and size > MIN_SIZE:
            size = max(size - 4, MIN_SIZE)
            font = load_font(font_name, size)
            lines = wrap_lines(draw, text, font, max_width)
    return size


def remove_silence(audio_path: Path, tmp_path: Path) -> tuple[Path, list]:
    """Remove silence from audio and return (new_path, silence_intervals).
    silence_intervals: list of (start, end) in original timeline that were removed."""
    # Detect silence: gaps > 0.4s below -38dB
    detect = subprocess.run(
        ["ffmpeg", "-i", str(audio_path),
         "-af", "silencedetect=noise=-38dB:d=0.4",
         "-f", "null", "-"],
        capture_output=True, text=True, timeout=120
    )
    output = detect.stderr

    # Parse silence intervals
    intervals = []
    starts = []
    for line in output.splitlines():
        if "silence_start" in line:
            try:
                starts.append(float(line.split("silence_start:")[1].strip()))
            except Exception:
                pass
        elif "silence_end" in line and starts:
            try:
                parts = line.split("silence_end:")[1].strip().split("|")
                end = float(parts[0].strip())
                # Keep 0.15s of silence at the start of each gap (natural breath)
                gap_start = starts.pop(0) + 0.15
                if end - gap_start > 0.05:
                    intervals.append((gap_start, end))
            except Exception:
                pass

    if not intervals:
        return audio_path, []

    # Build FFmpeg atrim filter to cut out silence intervals
    # Strategy: keep all non-silent segments, concat them
    duration_probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
        capture_output=True, text=True
    )
    total_dur = float(duration_probe.stdout.strip())

    # Build segments to keep
    keep = []
    prev = 0.0
    for (s, e) in intervals:
        if s > prev:
            keep.append((prev, s))
        prev = e
    if prev < total_dur:
        keep.append((prev, total_dur))

    if not keep:
        return audio_path, intervals

    # Build filter_complex with atrim + concat
    filter_parts = []
    for i, (s, e) in enumerate(keep):
        filter_parts.append(f"[0:a]atrim={s:.3f}:{e:.3f},asetpts=PTS-STARTPTS[seg{i}]")
    concat_inputs = "".join(f"[seg{i}]" for i in range(len(keep)))
    filter_parts.append(f"{concat_inputs}concat=n={len(keep)}:v=0:a=1[aout]")

    clean_path = tmp_path / "audio_clean.mp3"
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path),
         "-filter_complex", ";".join(filter_parts),
         "-map", "[aout]", "-c:a", "libmp3lame", "-q:a", "2",
         str(clean_path)],
        capture_output=True, text=True, timeout=120
    )
    if r.returncode != 0:
        print(f"[silence remove] failed: {r.stderr[-500:]}", file=sys.stderr)
        return audio_path, []

    return clean_path, intervals


def adjust_timestamps(transcript_data: list, intervals: list) -> list:
    """Shift timestamps to account for removed silence intervals."""
    if not intervals:
        return transcript_data

    def shift(t: float) -> float:
        offset = 0.0
        for (s, e) in intervals:
            if t <= s:
                break
            if t >= e:
                offset += e - s
            else:
                offset += t - s
        return round(t - offset, 3)

    result = []
    for block in transcript_data:
        result.append({
            "text":  block["text"],
            "start": shift(block["start"]),
            "end":   shift(block["end"]),
        })
    return result


def _split_chunks(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        if start + max_chars >= len(text):
            chunks.append(text[start:].strip())
            break
        sub = text[start:start + max_chars]
        cut = -1
        for i in range(len(sub) - 1, max_chars // 2, -1):
            if sub[i] in '.!?' and i + 1 < len(sub) and sub[i + 1] == ' ':
                cut = i + 2
                break
        if cut < 0:
            sp = sub.rfind(' ')
            cut = sp + 1 if sp > 0 else max_chars
        chunks.append(text[start:start + cut].strip())
        start += cut
    return [c for c in chunks if c]


def _clean_text(text: str) -> str:
    clean = " ".join(text.split())
    for bad, good in [
        ("â", "'"), ("â", "“"), ("â", "”"),
        ("â", "—"), ("â", "–"), ("â", "—"),
    ]:
        clean = clean.replace(bad, good)
    return clean


@app.post("/tts")
async def text_to_speech(
    text: str = Form(...),
    voice_id: str = Form("cm1VTuOWsFQRdZ5uDzSB"),
    api_key: str = Form(...),
    speed: float = Form(1.1),
):
    clean = _clean_text(text)
    chunks = _split_chunks(clean, 1500)
    parts = []
    async with httpx.AsyncClient(timeout=60.0) as client:
        for i, chunk in enumerate(chunks):
            r = await client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={"xi-api-key": api_key, "Content-Type": "application/json"},
                json={
                    "text": chunk,
                    "model_id": "eleven_v3",
                    "speed": speed,
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "style": 0.0, "use_speaker_boost": True},
                },
            )
            if r.status_code != 200 or len(r.content) < 100:
                raise HTTPException(status_code=500, detail=f"ElevenLabs chunk {i+1}: {r.text[:300]}")
            parts.append(r.content)
    return Response(content=b"".join(parts), media_type="audio/mpeg")


def split_text_into_subtitle_blocks(text: str, client, max_chars: int = MAX_SUBTITLE_CHARS) -> list[str]:
    flat_text = " ".join(text.split())
    # Normalize dashes: em/en dash → hyphen (some fonts lack em-dash glyph)
    flat_text = flat_text.replace("—", " - ").replace("–", " - ")

    prompt = f"""You are a subtitle editor. Split the following text into subtitle blocks for a vertical video.

RULES:
1. Each block must be maximum {max_chars} characters (including spaces)
2. ALWAYS split at sentence endings (. ? ! ." .') — every sentence must be its own block, never combine two sentences into one block
3. If a sentence is longer than {max_chars} characters, split at a comma, em dash (— or –), or any other punctuation mark
4. Never cut mid-thought — each block must feel like a complete phrase or natural pause
5. Do NOT change, add, or remove any words — only split
6. Return ONLY the blocks, one per line, no numbering, no extra text

TEXT:
{flat_text}"""

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()
    gpt_blocks = [line.strip() for line in raw.splitlines() if line.strip()]

    blocks = []
    for block in gpt_blocks:
        parts = re.split(r'(?<=[.?!])["’‘\']?\s+', block)
        blocks.extend([p.strip() for p in parts if p.strip()])

    return blocks


@app.post("/split")
async def split_text(text: str = Form(...)):
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)
    blocks = split_text_into_subtitle_blocks(text, client)
    return {"blocks": blocks}


def render_frame(text: str, bg_rgb: tuple, text_rgb: tuple, font, bg_image_pil=None, disclaimer_lines=None, font_name="") -> "Image":
    from PIL import Image, ImageDraw
    # LT Carpet lacks em/en dash glyphs — render as hyphens
    if "ltcarpet" in font_name.lower().replace(" ", "").replace("-", ""):
        text = text.replace("—", "-").replace("–", "-")
    if bg_image_pil is not None:
        img = _fit_image(bg_image_pil, VIDEO_WIDTH, VIDEO_HEIGHT)
    else:
        img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), bg_rgb)
    draw = ImageDraw.Draw(img)

    if text.strip():
        lines = wrap_lines(draw, text, font, int(VIDEO_WIDTH * 0.72))
        lines = fix_typography(lines)
        line_h = int(font.size * 1.18)
        total_h = len(lines) * line_h
        y = (VIDEO_HEIGHT - total_h) // 2
        for line in lines:
            w = draw.textbbox((0, 0), line, font=font)[2]
            draw.text(((VIDEO_WIDTH - w) // 2, y), line, font=font, fill=text_rgb)
            y += line_h

    if disclaimer_lines:
        disc_font = load_font("montserrat regular", 18)
        overlay   = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ov_draw   = ImageDraw.Draw(overlay)
        disc_line_h = 26
        total_disc_h = len(disclaimer_lines) * disc_line_h
        y = VIDEO_HEIGHT - 400 - total_disc_h
        # Pick text color based on bg brightness; photo bg → white with stroke
        if bg_image_pil is not None:
            disc_fill   = (255, 255, 255, 140)
            disc_stroke = (0, 0, 0, 120)
            disc_stroke_w = 2
        else:
            lum = (bg_rgb[0] * 299 + bg_rgb[1] * 587 + bg_rgb[2] * 114) // 1000
            disc_fill   = (0, 0, 0, 150) if lum > 160 else (255, 255, 255, 140)
            disc_stroke = None
            disc_stroke_w = 0
        for line in disclaimer_lines:
            w = ov_draw.textbbox((0, 0), line, font=disc_font)[2]
            x = (VIDEO_WIDTH - w) // 2
            ov_draw.text((x, y), line, font=disc_font, fill=disc_fill,
                         stroke_width=disc_stroke_w, stroke_fill=disc_stroke)
            y += disc_line_h
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    return img


@app.post("/render")
async def render_video(
    audio:      UploadFile           = File(...),
    transcript: str                  = Form(...),
    chat_id:    Optional[str]        = Form(None),
    bg_color:   str                  = Form("#000000"),
    text_color: str                  = Form("#FFFFFF"),
    font:       str                  = Form("Montserrat"),
    font_size:  str                  = Form("80"),
    bold:       str                  = Form("1"),
    bg_music:   Optional[UploadFile] = File(None),
):
    font_size_int = int(font_size)
    transcript_data = json.loads(transcript)
    transcript_data = split_long_blocks(transcript_data, font)
    audio_data = await audio.read()
    music_data = await bg_music.read() if bg_music else None

    bg_rgb   = hex_to_rgb(bg_color   if bg_color.startswith("#")   else f"#{bg_color}")
    text_rgb = hex_to_rgb(text_color if text_color.startswith("#") else f"#{text_color}")

    # Find one consistent font size for all blocks
    all_texts = [b["text"] for b in transcript_data]
    global_size = fit_font_size(all_texts, font, font_size_int)
    pil_font = load_font(font, global_size)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path   = Path(tmp)
        audio_path = tmp_path / "audio.mp3"
        frames_dir = tmp_path / "frames"
        no_music   = tmp_path / "no_music.mp4"
        output     = tmp_path / "output.mp4"
        frames_dir.mkdir()

        audio_path.write_bytes(audio_data)
        music_path = None
        if music_data:
            music_path = tmp_path / "music.mp3"
            music_path.write_bytes(music_data)

        # Normalize audio loudness (fixes volume jumps between TTS chunks)
        norm_path = tmp_path / "audio_norm.mp3"
        rn = subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path),
             "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
             "-c:a", "libmp3lame", "-q:a", "2", str(norm_path)],
            capture_output=True, text=True, timeout=120
        )
        if rn.returncode == 0:
            audio_path = norm_path

        # Remove silence and adjust timestamps
        audio_path, silence_intervals = remove_silence(audio_path, tmp_path)
        transcript_data = adjust_timestamps(transcript_data, silence_intervals)

        # Audio duration
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
                capture_output=True, text=True
            )
            audio_duration = float(probe.stdout.strip())
        except Exception:
            audio_duration = transcript_data[-1]["end"] if transcript_data else 0

        # Generate PNG frames (one per subtitle block)
        frame_paths = {}
        for idx, block in enumerate(transcript_data):
            fp = frames_dir / f"frame_{idx:04d}.png"
            render_frame(block["text"], bg_rgb, text_rgb, pil_font).save(str(fp), "PNG")
            frame_paths[idx] = fp

        # Build concat list — each block holds until next block starts (no black screen)
        concat_lines = []
        for idx, block in enumerate(transcript_data):
            dur = round(
                (transcript_data[idx + 1]["start"] if idx + 1 < len(transcript_data) else audio_duration)
                - block["start"], 3
            )
            if dur <= 0:
                continue
            concat_lines += [f"file '{frame_paths[idx]}'", f"duration {dur}"]
        if concat_lines:
            concat_lines.append(concat_lines[-2])  # repeat last (ffmpeg concat requirement)

        concat_file = tmp_path / "concat.txt"
        concat_file.write_text("\n".join(concat_lines))

        # Step 1: render frames + audio (Premiere-compatible settings)
        cmd1 = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-i", str(audio_path),
            "-vsync", "cfr", "-r", "30",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-profile:v", "baseline", "-level", "4.0",
            "-crf", "23",
            "-g", "30", "-keyint_min", "30", "-sc_threshold", "0",
            "-pix_fmt", "yuv420p",
            "-vf", "setsar=1:1",
            "-movflags", "+faststart",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-shortest",
            str(no_music),
        ]

        r1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=600)
        if r1.returncode != 0:
            print(f"[FFmpeg] rc={r1.returncode}\n{r1.stderr}", file=sys.stderr, flush=True)
            raise HTTPException(status_code=500,
                detail=f"rc={r1.returncode}\n{r1.stderr.strip()[-2000:]}")

        # Step 2: mix background music (optional)
        if music_path:
            cmd2 = [
                "ffmpeg", "-y",
                "-i", str(no_music),
                "-stream_loop", "-1", "-i", str(music_path),
                "-filter_complex",
                "[1:a]volume=0.1[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]",
                "-map", "0:v", "-map", "[aout]",
                "-c:v", "copy", "-c:a", "aac", "-ar", "48000", "-ac", "2",
                "-shortest", str(output),
            ]
            r2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=600)
            if r2.returncode != 0:
                print(f"[FFmpeg music] rc={r2.returncode}\n{r2.stderr}", file=sys.stderr, flush=True)
                shutil.copy(str(no_music), str(output))  # fallback: no music
        else:
            shutil.copy(str(no_music), str(output))

        video_bytes = output.read_bytes()

    return Response(
        content=video_bytes,
        media_type="video/mp4",
        headers={
            "X-Chat-Id": chat_id or "",
            "Content-Disposition": 'attachment; filename="video.mp4"',
        },
    )


def align_timestamps_python(gpt_blocks: list, words: list) -> list:
    def norm(w: str) -> str:
        return re.sub(r'[^a-zA-Z0-9]', '', w).lower()

    if not words:
        return []

    w_norm = [norm(w['word']) for w in words]
    n = len(w_norm)
    word_pos: dict = {}
    for i, w in enumerate(w_norm):
        if w:
            word_pos.setdefault(w, []).append(i)

    # MAX_GAP: if the matched position jumps more than this many seconds from the
    # previous matched block, it's treated as a "suspicious" match (e.g. GPT reordered
    # a bridge/recap section before the body, but TTS speaks it later in the audio).
    # Suspicious matches are still recorded with their real timestamps so the subtitles
    # sync to the audio, but cursor does NOT advance — letting body blocks find their
    # earlier positions via the fallback search range [cursor, sus_cursor).
    MAX_GAP = 20.0

    # Phase 1: try to align each block, record success/failure separately
    raw: list[tuple] = []   # (text, start_time, end_time, aligned: bool)
    cursor = 0       # conservative: advances only on non-suspicious matches
    sus_cursor = 0   # always advances; marks the high-water-mark of matched positions
    prev_end_time = 0.0  # only updated on non-suspicious matches

    for block in gpt_blocks:
        b_words = block.split()
        b_norm = [norm(w) for w in b_words if norm(w)]
        if not b_norm:
            continue
        start_idx = None
        is_sus = False
        best_sus = None   # best suspicious candidate from primary search

        # ── Primary search: from max(cursor, sus_cursor) forward ──────────────
        search_from = max(cursor, sus_cursor)
        for off in range(min(3, len(b_norm))):
            if start_idx is not None:
                break
            for pos in [p for p in word_pos.get(b_norm[off], []) if p - off >= search_from]:
                cs = pos - off
                matches = sum(1 for d, bw in enumerate(b_norm[:4]) if cs + d < n and w_norm[cs + d] == bw)
                if matches >= (1 if len(b_norm) <= 2 else 2):
                    gap = words[cs]['start'] - prev_end_time if prev_end_time > 0 else 0.0
                    if gap <= MAX_GAP:
                        start_idx = cs; is_sus = False; break
                    elif best_sus is None:
                        best_sus = cs

        # ── Fallback: if primary failed and sus_cursor skipped body content,
        #    search in [cursor, sus_cursor) for a non-suspicious match ──────────
        if start_idx is None and sus_cursor > cursor:
            for off in range(min(3, len(b_norm))):
                if start_idx is not None:
                    break
                for pos in [p for p in word_pos.get(b_norm[off], []) if cursor <= p - off < sus_cursor]:
                    cs = pos - off
                    matches = sum(1 for d, bw in enumerate(b_norm[:4]) if cs + d < n and w_norm[cs + d] == bw)
                    if matches >= (1 if len(b_norm) <= 2 else 2):
                        gap = words[cs]['start'] - prev_end_time if prev_end_time > 0 else 0.0
                        if gap <= MAX_GAP:
                            start_idx = cs; is_sus = False; break

        # ── If still no normal match, take the suspicious one ─────────────────
        if start_idx is None and best_sus is not None:
            start_idx = best_sus; is_sus = True

        # ── Look-back (unchanged) ─────────────────────────────────────────────
        if start_idx is None:
            lb = max(0, cursor - 10)
            for off in range(min(3, len(b_norm))):
                if start_idx is not None:
                    break
                for pos in sorted([p for p in word_pos.get(b_norm[off], []) if lb <= p - off < cursor], reverse=True):
                    cs = pos - off
                    matches = sum(1 for d, bw in enumerate(b_norm[:4]) if cs + d < n and w_norm[cs + d] == bw)
                    if matches >= (1 if len(b_norm) <= 2 else 2):
                        start_idx = cs
                        break

        fixed = re.sub(r'\b(Releyshio|RelayShow|Relay\s*Show|Rilaysho)\b', 'Relatio', block, flags=re.IGNORECASE)
        if start_idx is not None:
            end_idx = min(start_idx + len(b_words) - 1, n - 1)
            t_start = round(words[start_idx]['start'], 3)
            t_end   = round(words[end_idx]['end'], 3)
            raw.append((fixed, t_start, t_end, True))
            sus_cursor = max(sus_cursor, end_idx + 1)  # always advance sus_cursor
            if is_sus:
                # cursor held — body blocks can still search behind sus_cursor
                print(f"[align] suspicious '{block[:30]}' t={t_start:.1f} gap={t_start-prev_end_time:.1f}s cursor_held={cursor}", flush=True)
            else:
                cursor = max(cursor, end_idx + 1)
                prev_end_time = t_end
        else:
            key = b_norm[0] if b_norm else ''
            near = [(p, round(words[p]['start'], 2)) for p in word_pos.get(key, []) if abs(p - cursor) < 20][:4]
            print(f"[align] fail '{block[:30]}' cursor={cursor} key='{key}' near={near}", flush=True)
            raw.append((fixed, None, None, False))

    if not raw:
        return []

    # Phase 2: interpolate timestamps for failed blocks so nothing gets swallowed
    total_dur = words[-1]['end']

    # Build per-index "nearest known end before" and "nearest known start after"
    prev_end = [0.0] * len(raw)
    last_end = 0.0
    for i, (_, ts, te, ok) in enumerate(raw):
        prev_end[i] = last_end
        if ok:
            last_end = te

    next_start = [total_dur] * len(raw)
    nxt = total_dur
    for i in range(len(raw) - 1, -1, -1):
        _, ts, te, ok = raw[i]
        next_start[i] = nxt
        if ok:
            nxt = ts

    result = []
    i = 0
    while i < len(raw):
        text, ts, te, ok = raw[i]
        if ok:
            result.append({'text': text, 'start': ts, 'end': te})
            i += 1
        else:
            # Collect the full run of consecutive failures
            j = i
            while j < len(raw) and not raw[j][3]:
                j += 1
            run_texts = [raw[k][0] for k in range(i, j)]
            p = prev_end[i]
            q = next_start[i]
            step = (q - p) / (len(run_texts) + 1)
            for k, rt in enumerate(run_texts):
                t = round(p + (k + 1) * step, 3)
                result.append({'text': rt, 'start': t, 'end': round(t + step * 0.9, 3)})
            i = j

    ok_count = sum(1 for _, _, _, ok in raw if ok)
    fail_examples = [(i, raw[i][0][:35], [norm(w) for w in raw[i][0].split() if norm(w)][:1])
                     for i in range(len(raw)) if not raw[i][3]][:5]
    ok_examples   = [(i, round(raw[i][1],2), raw[i][0][:35])
                     for i in range(len(raw)) if raw[i][3]][:9]
    print(f"[align] blocks={len(raw)} words={n} ok={ok_count} fail={len(raw)-ok_count} "
          f"ok_blocks={ok_examples} "
          f"fail_examples={fail_examples}", flush=True)
    return result


FONT_SIZES = {
    'montserrat': 58, 'gilroy': 56, 'georgia': 54,
    'ltcarpet': 52, 'inter': 58, 'bodyhand': 50,
}


@app.post("/generate")
async def generate_video_web(
    text:       str           = Form(...),
    voice_id:   str           = Form("cm1VTuOWsFQRdZ5uDzSB"),
    font:       str           = Form("Montserrat"),
    bg_color:   str           = Form("#000000"),
    text_color: str           = Form("#FFFFFF"),
    font_size:  str           = Form("0"),
    music:                Optional[UploadFile] = File(None),
    preset_music:         str = Form(""),
    bg_image:             Optional[UploadFile] = File(None),
    use_christmas_frame:  str = Form("0"),
    frame_file:           str = Form("christmas.png"),
    disclaimer:           str = Form(""),
):
    el_key  = os.environ.get("ELEVENLABS_API_KEY", "")
    oai_key = os.environ.get("OPENAI_API_KEY", "")
    if not el_key:
        raise HTTPException(500, "ELEVENLABS_API_KEY not set")
    if not oai_key:
        raise HTTPException(500, "OPENAI_API_KEY not set")

    requested = int(font_size) if font_size.isdigit() and int(font_size) > 0 else 0
    font_size_int = requested if requested >= 36 else FONT_SIZES.get(font.lower(), 58)

    bg_image_data = await bg_image.read() if bg_image else None

    bg_image_pil = None
    if bg_image_data:
        from PIL import Image
        import io
        bg_image_pil = Image.open(io.BytesIO(bg_image_data))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # 1. TTS with-timestamps (returns audio + word timing, no separate Whisper needed)
        clean = _clean_text(text)
        chunks = _split_chunks(clean, 1500)
        audio_parts = []
        all_words: list[dict] = []
        time_offset = 0.0
        async with httpx.AsyncClient(timeout=90.0) as client:
            for i, chunk in enumerate(chunks):
                r = await client.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps",
                    headers={"xi-api-key": el_key, "Content-Type": "application/json"},
                    json={"text": chunk, "model_id": "eleven_v3", "speed": 1.1,
                          "voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "style": 0.0, "use_speaker_boost": True}},
                )
                if r.status_code != 200:
                    raise HTTPException(500, f"TTS error chunk {i+1}: {r.text[:200]}")
                data = r.json()
                chunk_audio = base64.b64decode(data["audio_base64"])
                if len(chunk_audio) < 100:
                    raise HTTPException(500, f"TTS error chunk {i+1}: audio too short")
                audio_parts.append(chunk_audio)

                alignment  = data.get("alignment", {})
                chars      = alignment.get("characters", [])
                char_starts = alignment.get("character_start_times_seconds", [])
                char_ends   = alignment.get("character_end_times_seconds", [])

                j = 0
                while j < len(chars):
                    if chars[j] in (" ", "\n", "\t"):
                        j += 1
                        continue
                    k = j
                    while k < len(chars) and chars[k] not in (" ", "\n", "\t"):
                        k += 1
                    if k > j and j < len(char_starts) and k - 1 < len(char_ends):
                        all_words.append({
                            "word":  "".join(chars[j:k]),
                            "start": round(char_starts[j] + time_offset, 3),
                            "end":   round(char_ends[k - 1] + time_offset, 3),
                        })
                    j = k

                # Probe actual MP3 duration for accurate time_offset (trailing silence
                # makes char_ends[-1] shorter than the real audio length)
                chunk_file = tmp_path / f"chunk_{i}.mp3"
                chunk_file.write_bytes(chunk_audio)
                dur_probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(chunk_file)],
                    capture_output=True, text=True,
                )
                try:
                    time_offset += float(dur_probe.stdout.strip())
                except Exception:
                    if char_ends:
                        time_offset += char_ends[-1]

        audio_path = tmp_path / "audio.mp3"
        audio_path.write_bytes(b"".join(audio_parts))

        # 2. Loudnorm
        norm_path = tmp_path / "audio_norm.mp3"
        rn = subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path), "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
             "-c:a", "libmp3lame", "-q:a", "2", str(norm_path)],
            capture_output=True, text=True, timeout=120
        )
        if rn.returncode == 0:
            audio_path = norm_path

        # 3. Remove silence; adjust EL word timestamps to match cleaned audio
        audio_path, silence_intervals = remove_silence(audio_path, tmp_path)
        if silence_intervals:
            adjusted = adjust_timestamps(
                [{"text": w["word"], "start": w["start"], "end": w["end"]} for w in all_words],
                silence_intervals,
            )
            all_words = [{"word": b["text"], "start": b["start"], "end": b["end"]} for b in adjusted]

        # 4. GPT subtitle split (uses original clean text — no Whisper transcription needed)
        from openai import OpenAI
        oai = OpenAI(api_key=oai_key)
        gpt_blocks = split_text_into_subtitle_blocks(clean, oai)

        # 5. Align GPT blocks with EL word timestamps
        transcript_data = align_timestamps_python(gpt_blocks, all_words)
        transcript_data = split_long_blocks(transcript_data, font, font_size_int)
        # Sort by start time: alignment may produce out-of-order blocks (e.g. when GPT
        # reorders a bridge section before the body but TTS speaks it later).
        # Frame rendering uses next_block["start"] as end time, so list order must match
        # chronological order to avoid 70-second frames and skipped body blocks.
        transcript_data = sorted(transcript_data, key=lambda b: b["start"])

        # 7. Render
        bg_rgb   = hex_to_rgb(bg_color   if bg_color.startswith("#")   else f"#{bg_color}")
        text_rgb = hex_to_rgb(text_color if text_color.startswith("#") else f"#{text_color}")
        pil_font = load_font(font, font_size_int)
        disclaimer_lines = [l.strip() for l in disclaimer.split("\n") if l.strip()]

        if preset_music:
            preset_path = MUSIC_DIR / preset_music
            music_data = preset_path.read_bytes() if preset_path.exists() else None
        else:
            music_data = await music.read() if music else None
        frames_dir = tmp_path / "frames"
        no_music   = tmp_path / "no_music.mp4"
        output     = tmp_path / "output.mp4"
        frames_dir.mkdir()

        music_path = None
        if music_data:
            music_path = tmp_path / "music.mp3"
            music_path.write_bytes(music_data)

        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
                capture_output=True, text=True)
            audio_duration = float(probe.stdout.strip())
        except Exception:
            audio_duration = transcript_data[-1]["end"] if transcript_data else 0

        frame_paths = {}
        for idx, block in enumerate(transcript_data):
            fp = frames_dir / f"frame_{idx:04d}.png"
            render_frame(block["text"], bg_rgb, text_rgb, pil_font, bg_image_pil=bg_image_pil, disclaimer_lines=disclaimer_lines or None, font_name=font).save(str(fp), "PNG")
            frame_paths[idx] = fp

        concat_lines = []
        for idx, block in enumerate(transcript_data):
            dur = round((transcript_data[idx+1]["start"] if idx+1 < len(transcript_data) else audio_duration) - block["start"], 3)
            if dur <= 0:
                continue
            concat_lines += [f"file '{frame_paths[idx]}'", f"duration {dur}"]
        if concat_lines:
            concat_lines.append(concat_lines[-2])

        concat_file = tmp_path / "concat.txt"
        concat_file.write_text("\n".join(concat_lines))

        cmd1 = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file),
                "-i", str(audio_path), "-vsync", "cfr", "-r", "30",
                "-c:v", "libx264", "-preset", "ultrafast", "-profile:v", "baseline", "-level", "4.0",
                "-crf", "23", "-g", "30", "-keyint_min", "30", "-sc_threshold", "0",
                "-pix_fmt", "yuv420p", "-vf", "setsar=1:1", "-movflags", "+faststart",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-shortest", str(no_music)]
        r1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=600)
        if r1.returncode != 0:
            raise HTTPException(500, f"FFmpeg: {r1.stderr[-1000:]}")

        if music_path:
            cmd2 = ["ffmpeg", "-y", "-i", str(no_music), "-stream_loop", "-1", "-i", str(music_path),
                    "-filter_complex", "[1:a]volume=0.1[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]",
                    "-map", "0:v", "-map", "[aout]", "-c:v", "copy",
                    "-c:a", "aac", "-ar", "48000", "-ac", "2", "-shortest", str(output)]
            r2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=600)
            if r2.returncode != 0:
                shutil.copy(str(no_music), str(output))
        else:
            shutil.copy(str(no_music), str(output))

        # Optional: overlay Christmas lights frame
        if use_christmas_frame == "1":
            safe_frame = Path(frame_file).name
            frame_src = FRAMES_DIR / safe_frame
            if frame_src.exists():
                framed = tmp_path / "output_framed.mp4"
                cmd_frame = [
                    "ffmpeg", "-y",
                    "-i", str(output),
                    "-i", str(frame_src),
                    "-filter_complex",
                    f"[1:v]scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}[fr];[0:v][fr]overlay=0:0",
                    "-c:v", "libx264", "-preset", "ultrafast",
                    "-crf", "23", "-pix_fmt", "yuv420p",
                    "-c:a", "copy",
                    str(framed),
                ]
                rf = subprocess.run(cmd_frame, capture_output=True, text=True, timeout=300)
                if rf.returncode == 0:
                    output = framed

        video_bytes = output.read_bytes()
        audio_bytes = audio_path.read_bytes()

        # TXT — continuous text, no paragraph breaks
        txt_content = " ".join(text.split())

        # SRT — seamless: each block runs until the next one starts (no gaps)
        def fmt_time(s: float) -> str:
            ms = int(round(s * 1000))
            h, ms = divmod(ms, 3_600_000)
            m, ms = divmod(ms, 60_000)
            sec, ms = divmod(ms, 1000)
            return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

        srt_lines = []
        for i, block in enumerate(transcript_data):
            end_t = transcript_data[i + 1]["start"] if i + 1 < len(transcript_data) else audio_duration
            srt_lines.append(str(i + 1))
            srt_lines.append(f"{fmt_time(block['start'])} --> {fmt_time(end_t)}")
            srt_lines.append(block["text"])
            srt_lines.append("")
        srt_content = "\n".join(srt_lines)

        bg_label    = "Photo" if bg_image_data else bg_color
        frame_label = frame_file.replace(".png", "") if use_christmas_frame == "1" else "None"
        music_label = (preset_music.replace(".mp3", "").replace(".m4a", "") if preset_music
                       else ("Custom file" if music_data else "None"))
        disc_label  = disclaimer.replace("\n", " | ") if disclaimer.strip() else "None"
        settings_content = (
            f"Font: {font}\n"
            f"Font Size: {font_size_int}\n"
            f"Text Color: {text_color}\n"
            f"Background: {bg_label}\n"
            f"Frame: {frame_label}\n"
            f"Disclaimer: {disc_label}\n"
            f"Music: {music_label}\n"
        )

    return JSONResponse({
        "video":    base64.b64encode(video_bytes).decode(),
        "audio":    base64.b64encode(audio_bytes).decode(),
        "txt":      txt_content,
        "srt":      srt_content,
        "settings": settings_content,
    })


# ─── Async generation (SSE log) ──────────────────────────────────────────────

def _generate_core_sync(log, params: dict) -> dict:
    """Sync version of the generate pipeline. Called from background thread."""
    el_key              = params["el_key"]
    oai_key             = params["oai_key"]
    text                = params["text"]
    voice_id            = params["voice_id"]
    font                = params["font"]
    bg_color            = params["bg_color"]
    text_color          = params["text_color"]
    font_size_int       = params["font_size_int"]
    music_data          = params.get("music_data")
    preset_music        = params.get("preset_music", "")
    bg_image_data       = params.get("bg_image_data")
    use_christmas_frame = params.get("use_christmas_frame", "0")
    frame_file          = params.get("frame_file", "christmas.png")
    disclaimer          = params.get("disclaimer", "")

    bg_image_pil = None
    if bg_image_data:
        from PIL import Image
        import io as _io
        bg_image_pil = Image.open(_io.BytesIO(bg_image_data))

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # 1. TTS with-timestamps (sync via requests)
        clean  = _clean_text(text)
        chunks = _split_chunks(clean, 1500)
        audio_parts: list[bytes] = []
        all_words: list[dict]    = []
        time_offset = 0.0

        log(f"🎙 TTS: {len(chunks)} чанк{'и' if len(chunks) > 1 else ''}…")
        for i, chunk in enumerate(chunks):
            r = _req.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/with-timestamps",
                headers={"xi-api-key": el_key, "Content-Type": "application/json"},
                json={"text": chunk, "model_id": "eleven_v3", "speed": 1.1,
                      "voice_settings": {"stability": 0.5, "similarity_boost": 0.75,
                                         "style": 0.0, "use_speaker_boost": True}},
                timeout=90,
            )
            if r.status_code != 200:
                raise ValueError(f"TTS error chunk {i+1}: {r.text[:200]}")
            data = r.json()
            chunk_audio = base64.b64decode(data["audio_base64"])
            if len(chunk_audio) < 100:
                raise ValueError(f"TTS error chunk {i+1}: audio too short")
            audio_parts.append(chunk_audio)

            alignment   = data.get("alignment", {})
            chars       = alignment.get("characters", [])
            char_starts = alignment.get("character_start_times_seconds", [])
            char_ends   = alignment.get("character_end_times_seconds", [])

            j = 0
            while j < len(chars):
                if chars[j] in (" ", "\n", "\t"):
                    j += 1; continue
                k = j
                while k < len(chars) and chars[k] not in (" ", "\n", "\t"):
                    k += 1
                if k > j and j < len(char_starts) and k - 1 < len(char_ends):
                    all_words.append({
                        "word":  "".join(chars[j:k]),
                        "start": round(char_starts[j] + time_offset, 3),
                        "end":   round(char_ends[k - 1] + time_offset, 3),
                    })
                j = k

            chunk_file = tmp_path / f"chunk_{i}.mp3"
            chunk_file.write_bytes(chunk_audio)
            dur_probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(chunk_file)],
                capture_output=True, text=True,
            )
            try:
                time_offset += float(dur_probe.stdout.strip())
            except Exception:
                if char_ends:
                    time_offset += char_ends[-1]

        audio_path = tmp_path / "audio.mp3"
        audio_path.write_bytes(b"".join(audio_parts))
        log("✅ Голос готовий")

        # 2. Loudnorm
        log("🔊 Нормалізація гучності…")
        norm_path = tmp_path / "audio_norm.mp3"
        rn = subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path),
             "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
             "-c:a", "libmp3lame", "-q:a", "2", str(norm_path)],
            capture_output=True, text=True, timeout=120,
        )
        if rn.returncode == 0:
            audio_path = norm_path

        # 3. Remove silence
        log("✂️ Видалення пауз…")
        audio_path, silence_intervals = remove_silence(audio_path, tmp_path)
        if silence_intervals:
            total_removed = sum(e - s for s, e in silence_intervals)
            print(f"[silence] removed {len(silence_intervals)} intervals, {total_removed:.2f}s total: {[(round(s,2),round(e,2)) for s,e in silence_intervals[:5]]}", flush=True)
            adjusted = adjust_timestamps(
                [{"text": w["word"], "start": w["start"], "end": w["end"]} for w in all_words],
                silence_intervals,
            )
            all_words = [{"word": b["text"], "start": b["start"], "end": b["end"]} for b in adjusted]

        # 4. GPT subtitle split
        log("📝 Розбивка на субтитри…")
        from openai import OpenAI
        oai = OpenAI(api_key=oai_key)
        gpt_blocks = split_text_into_subtitle_blocks(clean, oai)

        # 5. Align
        transcript_data = align_timestamps_python(gpt_blocks, all_words)
        transcript_data = split_long_blocks(transcript_data, font, font_size_int)
        transcript_data = sorted(transcript_data, key=lambda b: b["start"])
        log(f"✅ {len(transcript_data)} субтитр-блоків")

        # 6. Prepare render
        bg_rgb   = hex_to_rgb(bg_color   if bg_color.startswith("#")   else f"#{bg_color}")
        text_rgb = hex_to_rgb(text_color if text_color.startswith("#") else f"#{text_color}")
        pil_font = load_font(font, font_size_int)
        disclaimer_lines = [l.strip() for l in disclaimer.split("\n") if l.strip()]

        if preset_music:
            preset_path_obj = MUSIC_DIR / preset_music
            music_data = preset_path_obj.read_bytes() if preset_path_obj.exists() else None

        frames_dir = tmp_path / "frames"
        no_music   = tmp_path / "no_music.mp4"
        output     = tmp_path / "output.mp4"
        frames_dir.mkdir()

        music_path = None
        if music_data:
            music_path = tmp_path / "music.mp3"
            music_path.write_bytes(music_data)

        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path)],
                capture_output=True, text=True)
            audio_duration = float(probe.stdout.strip())
        except Exception:
            audio_duration = transcript_data[-1]["end"] if transcript_data else 0

        # 7. Render frames
        log(f"🖼 Рендеримо {len(transcript_data)} кадрів…")
        frame_paths = {}
        for idx, block in enumerate(transcript_data):
            fp = frames_dir / f"frame_{idx:04d}.png"
            render_frame(block["text"], bg_rgb, text_rgb, pil_font,
                         bg_image_pil=bg_image_pil,
                         disclaimer_lines=disclaimer_lines or None,
                         font_name=font).save(str(fp), "PNG")
            frame_paths[idx] = fp

        concat_lines = []
        for idx, block in enumerate(transcript_data):
            dur = round(
                (transcript_data[idx + 1]["start"] if idx + 1 < len(transcript_data) else audio_duration)
                - block["start"], 3
            )
            if dur <= 0:
                continue
            concat_lines += [f"file '{frame_paths[idx]}'", f"duration {dur}"]
        if concat_lines:
            concat_lines.append(concat_lines[-2])

        concat_file = tmp_path / "concat.txt"
        concat_file.write_text("\n".join(concat_lines))

        # 8. FFmpeg encode
        log("🎬 Кодуємо відео…")
        cmd1 = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-i", str(audio_path),
            "-vsync", "cfr", "-r", "30",
            "-c:v", "libx264", "-preset", "ultrafast", "-profile:v", "baseline", "-level", "4.0",
            "-crf", "23", "-g", "30", "-keyint_min", "30", "-sc_threshold", "0",
            "-pix_fmt", "yuv420p", "-vf", "setsar=1:1", "-movflags", "+faststart",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-shortest",
            str(no_music),
        ]
        r1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=600)
        if r1.returncode != 0:
            raise ValueError(f"FFmpeg: {r1.stderr[-800:]}")

        if music_path:
            log("🎵 Мікс з музикою…")
            cmd2 = [
                "ffmpeg", "-y", "-i", str(no_music),
                "-stream_loop", "-1", "-i", str(music_path),
                "-filter_complex",
                "[1:a]volume=0.1[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]",
                "-map", "0:v", "-map", "[aout]", "-c:v", "copy",
                "-c:a", "aac", "-ar", "48000", "-ac", "2", "-shortest", str(output),
            ]
            r2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=600)
            if r2.returncode != 0:
                shutil.copy(str(no_music), str(output))
        else:
            shutil.copy(str(no_music), str(output))

        if use_christmas_frame == "1":
            safe_frame = Path(frame_file).name
            frame_src  = FRAMES_DIR / safe_frame
            if frame_src.exists():
                framed   = tmp_path / "output_framed.mp4"
                cmd_frame = [
                    "ffmpeg", "-y", "-i", str(output), "-i", str(frame_src),
                    "-filter_complex",
                    f"[1:v]scale={VIDEO_WIDTH}:{VIDEO_HEIGHT}[fr];[0:v][fr]overlay=0:0",
                    "-c:v", "libx264", "-preset", "ultrafast",
                    "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "copy", str(framed),
                ]
                rf = subprocess.run(cmd_frame, capture_output=True, text=True, timeout=300)
                if rf.returncode == 0:
                    output = framed

        video_bytes = output.read_bytes()
        audio_bytes = audio_path.read_bytes()
        txt_content = " ".join(text.split())

        def fmt_time(s: float) -> str:
            ms = int(round(s * 1000))
            h, ms = divmod(ms, 3_600_000)
            m, ms = divmod(ms, 60_000)
            sec, ms = divmod(ms, 1000)
            return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

        srt_lines = []
        for i, block in enumerate(transcript_data):
            end_t = transcript_data[i + 1]["start"] if i + 1 < len(transcript_data) else audio_duration
            srt_lines += [str(i + 1),
                          f"{fmt_time(block['start'])} --> {fmt_time(end_t)}",
                          block["text"], ""]
        srt_content = "\n".join(srt_lines)

        bg_label    = "Photo" if bg_image_data else bg_color
        frame_label = frame_file.replace(".png", "") if use_christmas_frame == "1" else "None"
        music_label = (preset_music.replace(".mp3", "").replace(".m4a", "") if preset_music
                       else ("Custom file" if params.get("music_data") else "None"))
        disc_label  = disclaimer.replace("\n", " | ") if disclaimer.strip() else "None"
        settings_content = (
            f"Font: {font}\n"
            f"Font Size: {font_size_int}\n"
            f"Text Color: {text_color}\n"
            f"Background: {bg_label}\n"
            f"Frame: {frame_label}\n"
            f"Disclaimer: {disc_label}\n"
            f"Music: {music_label}\n"
        )

        # Save video to disk — avoids holding large bytes in RAM
        video_file = RESULTS_DIR / f"{params['_job_id']}.mp4"
        video_file.write_bytes(video_bytes)
        del video_bytes  # free immediately

        log("✅ Готово!")
        return {
            "audio":    base64.b64encode(audio_bytes).decode(),
            "txt":      txt_content,
            "srt":      srt_content,
            "settings": settings_content,
        }


def _run_generate(job_id: str, params: dict):
    if not _gen_sem.acquire(blocking=True, timeout=600):
        _jobs[job_id]["error"]  = "Queue timeout: server busy for too long"
        _jobs[job_id]["status"] = "error"
        _job_log(job_id, "❌ Queue timeout")
        return
    try:
        _job_log(job_id, "▶️ Починаємо…")
        result = _generate_core_sync(lambda msg: _job_log(job_id, msg), params)
        _jobs[job_id]["result"] = result
        _jobs[job_id]["status"] = "done"
    except Exception as e:
        _jobs[job_id]["error"]  = str(e)
        _jobs[job_id]["status"] = "error"
        _job_log(job_id, f"❌ {str(e)[:300]}")
    finally:
        _gen_sem.release()


@app.post("/generate/start")
async def generate_start(
    text:                 str           = Form(...),
    voice_id:             str           = Form("cm1VTuOWsFQRdZ5uDzSB"),
    font:                 str           = Form("Montserrat"),
    bg_color:             str           = Form("#000000"),
    text_color:           str           = Form("#FFFFFF"),
    font_size:            str           = Form("0"),
    music:                Optional[UploadFile] = File(None),
    preset_music:         str           = Form(""),
    bg_image:             Optional[UploadFile] = File(None),
    use_christmas_frame:  str           = Form("0"),
    frame_file:           str           = Form("christmas.png"),
    disclaimer:           str           = Form(""),
):
    el_key  = os.environ.get("ELEVENLABS_API_KEY", "")
    oai_key = os.environ.get("OPENAI_API_KEY", "")
    if not el_key:
        raise HTTPException(500, "ELEVENLABS_API_KEY not set")
    if not oai_key:
        raise HTTPException(500, "OPENAI_API_KEY not set")

    requested     = int(font_size) if font_size.isdigit() and int(font_size) > 0 else 0
    font_size_int = requested if requested >= 36 else FONT_SIZES.get(font.lower(), 58)

    bg_image_data = await bg_image.read() if bg_image else None
    music_data    = await music.read()    if music    else None

    job_id = secrets.token_hex(8)
    queued = not _gen_sem._value  # another job currently holds the semaphore
    init_logs = ["⏳ В черзі — зачекайте…"] if queued else []
    _jobs[job_id] = {"status": "running", "logs": init_logs, "result": None, "error": None}

    params = {
        "el_key": el_key, "oai_key": oai_key,
        "text": text, "voice_id": voice_id,
        "font": font, "bg_color": bg_color, "text_color": text_color,
        "font_size_int": font_size_int,
        "music_data": music_data, "preset_music": preset_music,
        "bg_image_data": bg_image_data,
        "use_christmas_frame": use_christmas_frame,
        "frame_file": frame_file, "disclaimer": disclaimer,
        "_job_id": job_id,
    }

    # Clean up result files older than 2 hours to free disk space
    try:
        cutoff = time.time() - 7200
        for f in RESULTS_DIR.glob("*.mp4"):
            if f.stat().st_mtime < cutoff:
                f.unlink(missing_ok=True)
    except Exception:
        pass

    threading.Thread(target=_run_generate, args=(job_id, params), daemon=True).start()
    return JSONResponse({"job_id": job_id})


@app.get("/events/{job_id}")
async def generation_events(job_id: str):
    async def stream():
        sent = 0
        tick = 0
        while True:
            job = _jobs.get(job_id)
            if not job:
                yield "data: ❌ Job not found\n\n"
                yield "data: __DONE__\n\n"
                return
            logs = job["logs"]
            while sent < len(logs):
                yield f"data: {logs[sent]}\n\n"
                sent += 1
            if job["status"] in ("done", "error"):
                yield "data: __DONE__\n\n"
                return
            tick += 1
            if tick % 30 == 0:          # keep-alive ping every ~9 s
                yield ": ping\n\n"
            await asyncio.sleep(0.3)
    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/result/{job_id}")
async def generation_result(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] == "running":
        return JSONResponse({"status": "running"})
    if job["status"] == "error":
        return JSONResponse({"status": "error", "error": job.get("error", "Unknown error")})
    result = job.pop("result")
    _jobs.pop(job_id, None)
    result["status"] = "done"
    # video is on disk — tell client to fetch it separately
    result["video_url"] = f"/download/{job_id}"
    return JSONResponse(result)


@app.get("/download/{job_id}")
async def download_video(job_id: str):
    path = RESULTS_DIR / f"{job_id}.mp4"
    if not path.exists():
        raise HTTPException(404, "Video not found or already downloaded")
    return FileResponse(
        str(path), media_type="video/mp4", filename="video.mp4",
        background=BackgroundTask(lambda: path.unlink(missing_ok=True))
    )


# ─── Scrolling Video ─────────────────────────────────────────────────────────

import uuid as _uuid
from scrolling_generator import generate_video as sv_generate_video, list_fonts as sv_list_fonts
from scrolling_tts import generate_sv_tts

_sv_tasks: dict[str, dict] = {}
SV_RESULTS_DIR = Path("/tmp/sv_results")
SV_UPLOADS_DIR = Path("/tmp/sv_uploads")
SV_RESULTS_DIR.mkdir(exist_ok=True)
SV_UPLOADS_DIR.mkdir(exist_ok=True)
_SV_FAVS_FILE = Path("/tmp/sv_favorites.json")


def _sv_daily_cleanup():
    """Delete scrolling video files older than 23 hours. Runs every 24 h."""
    while True:
        time.sleep(86400)
        cutoff = time.time() - 82800  # 23 h
        for d in (SV_RESULTS_DIR, SV_UPLOADS_DIR):
            try:
                for f in d.iterdir():
                    if f.is_file() and f.stat().st_mtime < cutoff:
                        f.unlink(missing_ok=True)
            except Exception:
                pass


threading.Thread(target=_sv_daily_cleanup, daemon=True).start()

_SV_VOICE_COLORS = ["#4f8ef7","#f75f4f","#4ff7a0","#f7d24f","#c24ff7",
                    "#f74fbe","#4ff7f7","#f7944f"]

def _sv_load_favs() -> set:
    try:
        import json as _json
        return set(_json.loads(_SV_FAVS_FILE.read_text()))
    except Exception:
        return set()

def _sv_save_favs(favs: set):
    import json as _json
    _SV_FAVS_FILE.write_text(_json.dumps(list(favs)))

def _sv_voice_color(voice_id: str) -> str:
    return _SV_VOICE_COLORS[hash(voice_id) % len(_SV_VOICE_COLORS)]


@app.get("/scrolling/fonts")
def sv_fonts():
    return JSONResponse(sv_list_fonts())


@app.get("/scrolling/music")
def sv_music_list():
    tracks = []
    for f in sorted(MUSIC_DIR.iterdir()):
        if f.suffix.lower() in (".mp3", ".wav", ".m4a", ".ogg"):
            tracks.append({"file": f.name, "name": f.stem})
    return JSONResponse(tracks)


@app.get("/scrolling/music/{filename}")
def sv_serve_music(filename: str):
    filename = Path(filename).name
    path = MUSIC_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(str(path))


@app.post("/scrolling/music/upload")
async def sv_upload_music(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No filename")
    ext = Path(file.filename).suffix.lower()
    if ext not in (".mp3", ".wav", ".m4a", ".ogg"):
        raise HTTPException(400, "Invalid file type")
    dest = MUSIC_DIR / Path(file.filename).name
    dest.write_bytes(await file.read())
    return JSONResponse({"file": dest.name, "name": dest.stem})


@app.post("/scrolling/upload/image")
async def sv_upload_image(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No filename")
    ext = Path(file.filename).suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        raise HTTPException(400, "Invalid file type")
    filename = str(_uuid.uuid4()) + ext
    dest = SV_UPLOADS_DIR / filename
    dest.write_bytes(await file.read())
    return JSONResponse({"filename": filename})


@app.get("/scrolling/voices")
def sv_voices():
    el_key = os.environ.get("ELEVENLABS_API_KEY", "")
    favs = _sv_load_favs()
    try:
        import httpx as _httpx
        r = _httpx.get("https://api.elevenlabs.io/v1/voices",
                       headers={"xi-api-key": el_key}, timeout=15)
        voices = r.json().get("voices", [])
        result = []
        for v in voices:
            result.append({
                "id":    v["voice_id"],
                "name":  v["name"],
                "fav":   v["voice_id"] in favs,
                "color": _sv_voice_color(v["voice_id"]),
            })
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse([])


@app.post("/scrolling/favorites/toggle")
async def sv_toggle_fav(request: Request):
    data = await request.json()
    vid = data.get("voice_id", "")
    favs = _sv_load_favs()
    if vid in favs:
        favs.discard(vid)
        is_fav = False
    else:
        favs.add(vid)
        is_fav = True
    _sv_save_favs(favs)
    return JSONResponse({"fav": is_fav})


def _sv_resolve_upload(filename: str) -> Optional[str]:
    if not filename:
        return None
    p = SV_UPLOADS_DIR / Path(filename).name
    return str(p) if p.exists() else None


def _sv_run(task_id: str, data: dict):
    if not _gen_sem.acquire(blocking=True, timeout=600):
        _sv_tasks[task_id] = {"status": "error", "progress": 0, "error": "Queue timeout"}
        return
    try:
        audio_path = None
        voice_id = data.get("voice_id")

        if voice_id:
            _sv_tasks[task_id] = {"status": "Генерую озвучку…", "progress": 10}
            audio_path = str(SV_RESULTS_DIR / f"{task_id}.mp3")
            tts_parts = []
            if data.get("title", "").strip():
                tts_parts.append(data["title"].strip())
            tts_parts.append(data["text"])
            tts_text = "\n\n".join(tts_parts)
            import re as _re
            tts_text = _re.sub(r'\bRelatio\b', 'Releyshio', tts_text, flags=_re.IGNORECASE)
            ok = generate_sv_tts(tts_text, voice_id, audio_path)
            if not ok:
                audio_path = None

        _sv_tasks[task_id] = {"status": "Рендерю відео…", "progress": 20}
        output = str(SV_RESULTS_DIR / f"{task_id}.mp4")

        music_file = data.get("music_file")
        music_path = str(MUSIC_DIR / Path(music_file).name) if music_file else None
        if music_path and not Path(music_path).exists():
            music_path = None

        sv_generate_video(
            title=data.get("title", ""),
            title_font_family=data.get("title_font_family", ""),
            title_font_size=int(data.get("title_font_size", 0)),
            title_color=data.get("title_color", ""),
            text=data["text"],
            bg_color=data.get("bg_color", "#0a0a0a"),
            text_color=data.get("text_color", "#ffffff"),
            font_family=data.get("font_family", "Arial Bold"),
            font_size=int(data.get("font_size", 40)),
            text_align=data.get("text_align", "left"),
            audio_path=audio_path,
            music_path=music_path,
            output_path=output,
            scroll_speed=int(data.get("scroll_speed", 40)),
            bg_image_path=_sv_resolve_upload(data.get("bg_image_file", "")),
            bg_image_opacity=float(data.get("bg_image_opacity", 0.5)),
            overlay_image_path=_sv_resolve_upload(data.get("overlay_image_file", "")),
            overlay_anchor_y=float(data.get("overlay_anchor_y", 0.5)),
            disclaimer=data.get("disclaimer", ""),
        )

        if audio_path and Path(audio_path).exists():
            Path(audio_path).unlink(missing_ok=True)

        # Read video into RAM so download works even if /tmp is cleared
        output_path_obj = Path(output)
        video_bytes = output_path_obj.read_bytes()
        output_path_obj.unlink(missing_ok=True)

        from datetime import datetime as _dt
        ts = _dt.now().strftime("%Y-%m-%d_%H-%M")
        _sv_tasks[task_id] = {
            "status": "done", "progress": 100,
            "filename": f"{task_id}.mp4",
            "display_filename": f"scrolling_{ts}.mp4",
            "video_bytes": video_bytes,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        _sv_tasks[task_id] = {"status": "error", "progress": 0, "error": str(e)}
    finally:
        _gen_sem.release()


@app.post("/scrolling/generate")
async def sv_generate(request: Request):
    data = await request.json()
    if not data or not data.get("text", "").strip():
        raise HTTPException(400, "text is required")
    task_id = str(_uuid.uuid4())
    queued = not _gen_sem._value
    _sv_tasks[task_id] = {"status": "⏳ В черзі…" if queued else "Починаємо…", "progress": 0}
    threading.Thread(target=_sv_run, args=(task_id, data), daemon=True).start()
    return JSONResponse({"task_id": task_id})


@app.get("/scrolling/status/{task_id}")
def sv_status(task_id: str):
    task = _sv_tasks.get(task_id)
    if not task:
        raise HTTPException(404, "Not found")
    return JSONResponse({k: v for k, v in task.items() if k != "video_bytes"})


@app.get("/scrolling/preview/{task_id}")
def sv_preview(task_id: str, request: Request):
    task = _sv_tasks.get(task_id)
    video_bytes = task.get("video_bytes") if task else None
    if not video_bytes:
        raise HTTPException(404, "Not found")
    total = len(video_bytes)
    range_header = request.headers.get("range")
    if range_header:
        try:
            ranges = range_header.strip().lower().replace("bytes=", "")
            start_str, end_str = ranges.split("-", 1)
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else total - 1
            end = min(end, total - 1)
            return Response(
                content=video_bytes[start:end + 1],
                status_code=206,
                media_type="video/mp4",
                headers={
                    "Content-Range": f"bytes {start}-{end}/{total}",
                    "Accept-Ranges": "bytes",
                    "Content-Length": str(end - start + 1),
                },
            )
        except Exception:
            pass
    return Response(
        content=video_bytes,
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes", "Content-Length": str(total)},
    )


@app.get("/scrolling/download/{task_id}")
def sv_download(task_id: str, name: str = ""):
    task = _sv_tasks.get(task_id)
    video_bytes = task.get("video_bytes") if task else None
    if not video_bytes:
        raise HTTPException(404, "Not found")
    display = Path(name).name if name else task.get("display_filename", f"{task_id}.mp4")
    return Response(
        content=video_bytes,
        media_type="video/mp4",
        headers={
            "Content-Disposition": f'attachment; filename="{display}"',
            "Content-Length": str(len(video_bytes)),
        },
    )


# ─── Voiceover Pipeline ──────────────────────────────────────────────────────

ELEVENLABS_API_KEY_VP = os.environ.get("ELEVENLABS_API_KEY", "")
OPENAI_API_KEY_VP     = os.environ.get("OPENAI_API_KEY", "")

VP_VOICE_COLORS = ["#4F8EF7","#A78BFA","#F472B6","#34D399","#FBBF24",
                   "#F87171","#38BDF8","#FB923C","#A3E635","#E879F9"]
VP_ELEVENLABS_MODEL  = "eleven_v3"
VP_VOICE_SETTINGS    = {"stability": 0.5, "similarity_boost": 0.75, "style": 0.0, "use_speaker_boost": True}
VP_OPENAI_MODEL      = "gpt-4o"
VP_CHUNK_SIZE        = 5000
VP_SILENCE_THRESHOLD = 0.10
VP_OUTPUT_DIR        = Path("/tmp/audio")
VP_FAV_FILE          = Path("/tmp/vp_favorites.json")

vp_voices:    list  = []
vp_favorites: set   = set()
vp_log_lines: list  = []
vp_is_running: bool = False
vp_last_file:  str  = None
vp_last_txt:   str  = None
vp_last_txt_name: str = "formatted.txt"
vp_stop_flag:  bool = False


def vp_load_favorites():
    global vp_favorites
    try:
        if VP_FAV_FILE.exists():
            vp_favorites = set(json.loads(VP_FAV_FILE.read_text()))
    except: pass

def vp_save_favorites():
    try: VP_FAV_FILE.write_text(json.dumps(list(vp_favorites), ensure_ascii=False))
    except: pass

def vp_fetch_voices():
    global vp_voices
    try:
        resp = _req.get("https://api.elevenlabs.io/v1/voices",
                        headers={"xi-api-key": ELEVENLABS_API_KEY_VP}, timeout=15)
        if resp.status_code != 200:
            return False
        data = resp.json().get("voices", [])
        data.sort(key=lambda v: (v.get("category","") not in ("cloned","professional"), v["name"].lower()))
        vp_voices = [
            {"id": v["voice_id"], "voice_id": v["voice_id"], "name": v["name"],
             "category": v.get("category",""), "color": VP_VOICE_COLORS[i % len(VP_VOICE_COLORS)]}
            for i, v in enumerate(data)
        ]
        return True
    except: return False

def vp_log(msg): vp_log_lines.append(msg + "\n")

def vp_format_text(text: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY_VP)
    vp_log("🔄 Форматуємо текст через ChatGPT...")
    resp = client.chat.completions.create(
        model=VP_OPENAI_MODEL,
        messages=[
            {"role": "system", "content": (
                "You are a text formatter. Apply these rules and return ONLY the result:\n"
                "1. Remove all paragraph breaks — join into a single continuous text.\n"
                "2. Ensure exactly one space before and after each long dash (– —). "
                "Short hyphens (-) stay unchanged.\n"
                "Return ONLY the formatted text."
            )},
            {"role": "user", "content": text}
        ],
        temperature=0.0
    )
    return resp.choices[0].message.content.strip()

def vp_replace_relatio(text: str) -> str:
    return re.sub(r'\bRelatio\b', 'Releyshio', text)

def vp_remove_silence(audio_path, output_path) -> bool:
    try:
        gradio = GradioClient("CDari/Remove-Silence-From-Audio_public2")
        result = gradio.predict(audio_file=gradio_handle_file(str(audio_path)),
                                seconds=VP_SILENCE_THRESHOLD, api_name="/predict")
        path = result[0] if isinstance(result, tuple) else result
        if path:
            shutil.move(str(path), str(output_path)); return True
        return False
    except Exception as e:
        vp_log(f"⚠️  Не вдалось видалити паузи: {e}"); return False

def vp_split_chunks(text: str) -> list:
    if len(text) <= VP_CHUNK_SIZE: return [text]
    pattern = re.compile(r'(?<=[.])\s+|(?<=[,])\s+|(?<=[–—])\s+')
    chunks, start = [], 0
    while start < len(text):
        rest = text[start:]
        if len(rest) <= VP_CHUNK_SIZE: chunks.append(rest.strip()); break
        sub = text[start: start + VP_CHUNK_SIZE]
        best = None
        for m in pattern.finditer(sub): best = m
        if best:
            end = start + best.end()
        else:
            sp = None
            for m in re.finditer(r'\s+', sub): sp = m
            end = start + (sp.end() if sp else VP_CHUNK_SIZE)
        chunks.append(text[start:end].strip()); start = end
    return [c for c in chunks if c]

def vp_tts_chunk(text: str, voice_id: str, path: Path) -> bool:
    resp = _req.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers={"xi-api-key": ELEVENLABS_API_KEY_VP, "Content-Type": "application/json"},
        json={"text": text, "model_id": VP_ELEVENLABS_MODEL, "voice_settings": VP_VOICE_SETTINGS},
        timeout=600)
    if resp.status_code == 200: path.write_bytes(resp.content); return True
    vp_log(f"❌ ElevenLabs {resp.status_code}: {resp.text[:200]}"); return False

def vp_run_pipeline(text: str, voice: dict, task_id: str = ""):
    global vp_is_running, vp_last_file, vp_last_txt, vp_last_txt_name, vp_stop_flag, vp_log_lines
    vp_is_running = True; vp_stop_flag = False
    vp_last_file = None; vp_last_txt = None
    vp_last_txt_name = "formatted.txt"; vp_log_lines = []
    try:
        vp_log(f"▶  Голос: {voice['name']}")
        vp_log(f"📝 Символів у тексті: {len(text)}\n")

        try:
            formatted = vp_format_text(text)
            vp_log(f"✅ Відформатовано ({len(formatted)} символів)")
            vp_last_txt = formatted
            vp_last_txt_name = f"{(task_id+'_') if task_id else ''}{voice['name']}_{VP_ELEVENLABS_MODEL.replace('-','_')}_formatted.txt".replace(' ','_')
        except Exception as e:
            vp_log(f"❌ Помилка форматування: {e}"); return

        if vp_stop_flag: vp_log("⛔ Зупинено"); return
        prepared = vp_replace_relatio(formatted)
        if prepared != formatted: vp_log("🔤 Замінено: Relatio → Releyshio")
        if vp_stop_flag: vp_log("⛔ Зупинено"); return

        VP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        task_part = f"{task_id}_" if task_id else ""
        filename = f"{task_part}{voice['name']}_{VP_ELEVENLABS_MODEL.replace('-','_')}_{int(time.time())}.mp3"
        filename = filename.replace(" ","_").replace("/","_")
        out_path = VP_OUTPUT_DIR / filename

        chunks = vp_split_chunks(prepared)
        if len(chunks) == 1:
            vp_log(f"\n🎙  Озвучуємо через ElevenLabs...")
            if not vp_tts_chunk(prepared, voice["voice_id"], out_path): return
        else:
            vp_log(f"\n📦 Текст розбито на {len(chunks)} частин")
            tmp_dir = VP_OUTPUT_DIR / "_tmp"; tmp_dir.mkdir(exist_ok=True)
            parts = []
            for i, chunk in enumerate(chunks, 1):
                if vp_stop_flag: vp_log("⛔ Зупинено"); return
                vp_log(f"  🎙  Чанк {i}/{len(chunks)} ({len(chunk)} символів)...")
                tmp = tmp_dir / f"chunk_{i:03d}.mp3"
                if not vp_tts_chunk(chunk, voice["voice_id"], tmp): return
                parts.append(tmp); time.sleep(0.5)
            vp_log("🔗 Склеюємо частини...")
            with open(out_path, "wb") as f:
                for p in parts: f.write(p.read_bytes())
            for p in parts: p.unlink(missing_ok=True)
            try: tmp_dir.rmdir()
            except: pass

        size_kb = out_path.stat().st_size // 1024
        vp_log(f"\n✅ Аудіо збережено ({size_kb} KB)")
        if vp_stop_flag: vp_log("⛔ Зупинено"); return

        vp_log(f"🔇 Видаляємо паузи (поріг {VP_SILENCE_THRESHOLD}с)...")
        clean_path = out_path.with_name(out_path.stem + "_clean.mp3")
        if GRADIO_AVAILABLE and vp_remove_silence(out_path, clean_path):
            size_kb = clean_path.stat().st_size // 1024
            vp_last_file = str(clean_path)
            vp_log(f"✅ Готово! ({size_kb} KB)")
        else:
            vp_last_file = str(out_path)
            vp_log(f"✅ Готово (без чистки пауз): {size_kb} KB")
    except Exception as e:
        vp_log(f"\n❌ Помилка: {e}")
    finally:
        vp_is_running = False


vp_load_favorites()
threading.Thread(target=vp_fetch_voices, daemon=True).start()


@app.get("/api/voices")
def vp_get_voices():
    return JSONResponse([{**v, "fav": v["id"] in vp_favorites} for v in vp_voices])

@app.get("/api/voices/refresh")
def vp_refresh_voices():
    vp_fetch_voices()
    return JSONResponse({"ok": True, "count": len(vp_voices)})

@app.get("/api/log")
def vp_get_log():
    return JSONResponse({"log": "".join(vp_log_lines), "running": vp_is_running,
                         "file": vp_last_file, "txt": bool(vp_last_txt)})

@app.get("/api/download")
def vp_download():
    if not vp_last_file: raise HTTPException(404, "No file")
    p = Path(vp_last_file)
    return Response(content=p.read_bytes(), media_type="audio/mpeg",
                    headers={"Content-Disposition": f'attachment; filename="{p.name}"'})

@app.get("/api/download_txt")
def vp_download_txt():
    if not vp_last_txt: raise HTTPException(404, "No txt")
    return Response(content=vp_last_txt.encode("utf-8"), media_type="text/plain; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{vp_last_txt_name}"'})

@app.post("/api/run")
async def vp_run(request: Request):
    if vp_is_running:
        return JSONResponse({"ok": False, "error": "Вже виконується"})
    body = await request.json()
    text = body.get("text", "").strip()
    if not text: return JSONResponse({"ok": False, "error": "Текст порожній"})
    voice = next((v for v in vp_voices if v["id"] == body.get("voice_id")), None)
    if not voice: return JSONResponse({"ok": False, "error": "Голос не знайдено"})
    task_id = body.get("task_id", "").strip().lstrip("#")
    threading.Thread(target=vp_run_pipeline, args=(text, voice, task_id), daemon=True).start()
    return JSONResponse({"ok": True})

@app.post("/api/stop")
async def vp_stop(request: Request):
    global vp_stop_flag
    vp_stop_flag = True
    return JSONResponse({"ok": True})

@app.post("/api/favorites/toggle")
async def vp_favorites_toggle(request: Request):
    body = await request.json()
    vid = body.get("voice_id", "")
    if vid in vp_favorites: vp_favorites.discard(vid)
    else: vp_favorites.add(vid)
    vp_save_favorites()
    return JSONResponse({"ok": True, "fav": vid in vp_favorites})


@app.get("/static-music/{filename}")
def serve_music_asset(filename: str):
    path = MUSIC_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Music not found")
    return Response(content=path.read_bytes(), media_type="audio/mpeg",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/static-frames/{filename}")
def serve_frame_asset(filename: str):
    path = FRAMES_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Frame not found")
    return Response(content=path.read_bytes(), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=86400"})


@app.get("/fonts/{filename}")
def serve_font(filename: str):
    path = FONT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Font not found")
    mime = "font/otf" if filename.endswith(".otf") else "font/truetype"
    return Response(content=path.read_bytes(), media_type=mime,
                    headers={"Cache-Control": "public, max-age=86400"})


def _serve_ui():
    p = Path(__file__).parent / "index.html"
    return HTMLResponse(
        content=p.read_text() if p.exists() else "<h1>Not found</h1>",
        headers={"Cache-Control": "no-store"},
    )

@app.get("/",               response_class=HTMLResponse)
def web_ui():          return _serve_ui()

@app.get("/easy-mh",        response_class=HTMLResponse)
def web_easy_mh():     return _serve_ui()

@app.get("/scrolling",      response_class=HTMLResponse)
def web_scrolling():   return _serve_ui()

@app.get("/voiceover",      response_class=HTMLResponse)
def web_voiceover():   return _serve_ui()

@app.get("/tiktok",         response_class=HTMLResponse)
def web_tiktok():      return _serve_ui()


# ── Text Cleaner (TikTok tab) ─────────────────────────────────────────────────

_TC_RULES_FILE = Path(__file__).parent / "tc_rules.json"
_TC_DEFAULT_RULES = [
    {"from": "get hard",          "to": "Get firm"},
    {"from": "hard",              "to": "Firm"},
    {"from": "sex",               "to": "Intimacy"},
    {"from": "masturb[a-z]*|masturnat[a-z]*", "to": "Beat off", "regex": True},
    {"from": "anxiety",           "to": "Stress"},
    {"from": "porn",              "to": "Spicy content"},
    {"from": "penis",             "to": "Member"},
    {"from": "pills",             "to": "Supplements"},
    {"from": "doctor",            "to": "Specialist"},
    {"from": "morning erections", "to": "morning wood"},
    {"from": "sexual position",   "to": "position"},
]

def _tc_load_rules() -> list:
    if _TC_RULES_FILE.exists():
        try:
            return json.loads(_TC_RULES_FILE.read_text("utf-8"))
        except Exception:
            pass
    return list(_TC_DEFAULT_RULES)

def _tc_save_rules(rules: list):
    _TC_RULES_FILE.write_text(json.dumps(rules, ensure_ascii=False, indent=2), "utf-8")

def _tc_apply_case(original: str, target: str) -> str:
    if not original or not target:
        return target
    if original.isupper() and len(original) > 1:
        return target.upper()
    if original[0].isupper():
        return target[0].upper() + target[1:]
    return target[0].lower() + target[1:]

def _tc_static_clean(text: str, rules: list) -> tuple[str, str]:
    if not rules:
        return text, _H.escape(text).replace("\n", "<br>")
    sorted_r = sorted(rules, key=lambda r: len(r["from"]), reverse=True)
    try:
        parts = []
        for r in sorted_r:
            if not r["from"]:
                continue
            pat = r["from"] if r.get("regex") else re.escape(r["from"])
            parts.append(r"(?<!\w)" + pat + r"(?!\w)")
        pattern = re.compile("|".join(parts), re.IGNORECASE)
    except re.error:
        return text, _H.escape(text).replace("\n", "<br>")
    plain_buf, diff_buf, last = [], [], 0
    for m in pattern.finditer(text):
        orig = m.group(0)
        repl = orig
        for r in sorted_r:
            if r.get("regex"):
                if re.fullmatch(r["from"], orig, re.IGNORECASE):
                    repl = _tc_apply_case(orig, r["to"]); break
            elif orig.lower() == r["from"].lower():
                repl = _tc_apply_case(orig, r["to"]); break
        before = text[last: m.start()]
        plain_buf.append(before)
        diff_buf.append(_H.escape(before).replace("\n", "<br>"))
        plain_buf.append(repl)
        diff_buf.append(f'<mark>{_H.escape(repl)}</mark>')
        last = m.end()
    tail = text[last:]
    plain_buf.append(tail)
    diff_buf.append(_H.escape(tail).replace("\n", "<br>"))
    return "".join(plain_buf), "".join(diff_buf)

def _tc_make_diff_html(original: str, cleaned: str) -> str:
    orig_t  = re.findall(r"\S+|\s+", original)
    clean_t = re.findall(r"\S+|\s+", cleaned)
    sm = difflib.SequenceMatcher(None, orig_t, clean_t, autojunk=False)
    parts = []
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        chunk = "".join(clean_t[j1:j2])
        parts.append(
            f'<mark>{_H.escape(chunk)}</mark>'
            if tag in ("replace", "insert") else _H.escape(chunk)
        )
    return "".join(parts).replace("\n", "<br>")

async def _tc_ai_clean(text: str, api_key: str) -> str:
    SYSTEM = (
        "You are a script editor. Lightly clean the text: fix grammar errors, "
        "improve sentence flow, align meaning where needed — but do NOT rewrite, "
        "do NOT change the core message, keep the same structure and approximate length. "
        "Return ONLY the corrected text, nothing else."
    )
    words = text.split()
    chunks = (
        [" ".join(words[i: i + 2500]) for i in range(0, len(words), 2500)]
        if len(words) > 2500 else [text]
    )
    results = []
    async with httpx.AsyncClient(timeout=120) as c:
        for chunk in chunks:
            r = await c.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o",
                    "temperature": 0.2,
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user",   "content": chunk},
                    ],
                },
            )
            r.raise_for_status()
            results.append(r.json()["choices"][0]["message"]["content"])
    return "\n".join(results)

@app.get("/tiktok/rules")
def tc_get_rules():
    return JSONResponse(_tc_load_rules())

@app.post("/tiktok/rules")
async def tc_set_rules(req: Request):
    _tc_save_rules(await req.json())
    return JSONResponse({"ok": True})

@app.post("/tiktok/clean")
async def tc_clean(req: Request):
    data    = await req.json()
    text    = data.get("text", "").strip()
    mode    = data.get("mode", "static")
    api_key = data.get("api_key", "") or os.environ.get("OPENAI_API_KEY", "")
    if not text:
        return JSONResponse({"error": "Empty text"}, status_code=400)
    rules = _tc_load_rules()
    cleaned, diff_html = _tc_static_clean(text, rules)
    if mode == "ai":
        if not api_key:
            return JSONResponse({"error": "OpenAI API key required. Open ⚙️ Settings."}, status_code=400)
        try:
            cleaned   = await _tc_ai_clean(cleaned, api_key)
            diff_html = _tc_make_diff_html(text, cleaned)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse({"cleaned": cleaned, "diff_html": diff_html})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/admin/cleanup")
def admin_cleanup():
    import gc
    freed_jobs = freed_tasks = freed_files = 0

    # Remove completed/errored jobs from memory
    done = [jid for jid, j in _jobs.items() if j.get("status") in ("done", "error")]
    for jid in done:
        _jobs.pop(jid, None)
        freed_jobs += 1

    # Remove completed/errored scrolling tasks from memory
    done_sv = [tid for tid, t in _sv_tasks.items() if t.get("status") in ("done", "error")]
    for tid in done_sv:
        _sv_tasks.pop(tid, None)
        freed_tasks += 1

    # Delete all mp4/mp3 result files
    for f in RESULTS_DIR.iterdir():
        try:
            f.unlink()
            freed_files += 1
        except Exception:
            pass

    # Delete leftover /tmp dirs from previous generations
    for d in Path("/tmp").iterdir():
        if d.is_dir() and d.name.startswith("tmp") and d != RESULTS_DIR:
            try:
                shutil.rmtree(d, ignore_errors=True)
                freed_files += 1
            except Exception:
                pass

    gc.collect()
    return JSONResponse({
        "freed_jobs": freed_jobs,
        "freed_tasks": freed_tasks,
        "freed_files": freed_files,
    })
