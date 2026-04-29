from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, HTMLResponse
from typing import Optional
import base64
import subprocess
import tempfile
import os
import json
import sys
import re
import httpx
from pathlib import Path
import shutil

app = FastAPI()

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

FONT_DIR = Path("/usr/share/fonts/truetype/montserrat")
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


def split_long_blocks(transcript_data: list, font_name: str) -> list:
    """Split blocks that still exceed MAX_LINES at minimum font size."""
    from PIL import Image, ImageDraw
    MAX_LINES = 4
    MIN_SIZE  = 36
    max_width = int(VIDEO_WIDTH * 0.72)
    draw = ImageDraw.Draw(Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT)))
    font = load_font(font_name, MIN_SIZE)

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

    prompt = f"""You are a subtitle editor. Split the following text into subtitle blocks for a vertical video.

RULES:
1. Each block must be maximum {max_chars} characters (including spaces)
2. Each block must be maximum 9 words — this is critical
3. Never put more than 9 words in one block
4. Prefer splitting at: sentence endings (. ? !), commas, or em dashes
5. Each block must feel like a complete thought or natural pause
6. Do NOT change, add, or remove any words - only split
7. Return ONLY the blocks, one per line, no numbering, no extra text

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


def render_frame(text: str, bg_rgb: tuple, text_rgb: tuple, font) -> "Image":
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), bg_rgb)
    if not text.strip():
        return img
    draw = ImageDraw.Draw(img)

    lines = wrap_lines(draw, text, font, int(VIDEO_WIDTH * 0.72))
    lines = fix_typography(lines)

    line_h = int(font.size * 1.18)
    total_h = len(lines) * line_h
    y = (VIDEO_HEIGHT - total_h) // 2

    for line in lines:
        w = draw.textbbox((0, 0), line, font=font)[2]
        draw.text(((VIDEO_WIDTH - w) // 2, y), line, font=font, fill=text_rgb)
        y += line_h

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
            "-c:v", "libx264", "-preset", "fast",
            "-profile:v", "high", "-level", "4.0",
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

    w_norm = [norm(w['word']) for w in words]
    n = len(w_norm)
    word_pos: dict = {}
    for i, w in enumerate(w_norm):
        if w:
            word_pos.setdefault(w, []).append(i)

    result, cursor = [], 0
    for block in gpt_blocks:
        b_norm = [norm(w) for w in block.split() if norm(w)]
        if not b_norm:
            continue
        start_idx = None
        for off in range(min(3, len(b_norm))):
            if start_idx is not None:
                break
            for pos in [p for p in word_pos.get(b_norm[off], []) if p - off >= cursor]:
                cs = pos - off
                matches = sum(1 for d, bw in enumerate(b_norm[:4]) if cs + d < n and w_norm[cs + d] == bw)
                if matches >= (1 if len(b_norm) <= 2 else 2):
                    start_idx = cs
                    break
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
        if start_idx is None:
            if result:
                result[-1]['text'] += ' ' + block
            continue
        end_idx = min(start_idx + len(b_norm) - 1, n - 1)
        fixed = re.sub(r'\b(Releyshio|RelayShow|Relay\s*Show|Rilaysho)\b', 'Relatio', block, flags=re.IGNORECASE)
        result.append({'text': fixed, 'start': round(words[start_idx]['start'], 3), 'end': round(words[end_idx]['end'], 3)})
        cursor = end_idx + 1
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
    music:      Optional[UploadFile] = File(None),
):
    el_key  = os.environ.get("ELEVENLABS_API_KEY", "")
    oai_key = os.environ.get("OPENAI_API_KEY", "")
    if not el_key:
        raise HTTPException(500, "ELEVENLABS_API_KEY not set")
    if not oai_key:
        raise HTTPException(500, "OPENAI_API_KEY not set")

    font_size_int = FONT_SIZES.get(font.lower(), 58)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # 1. TTS
        clean = _clean_text(text)
        chunks = _split_chunks(clean, 1500)
        parts = []
        async with httpx.AsyncClient(timeout=90.0) as client:
            for i, chunk in enumerate(chunks):
                r = await client.post(
                    f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                    headers={"xi-api-key": el_key, "Content-Type": "application/json"},
                    json={"text": chunk, "model_id": "eleven_v3", "speed": 1.1,
                          "voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "style": 0.0, "use_speaker_boost": True}},
                )
                if r.status_code != 200 or len(r.content) < 100:
                    raise HTTPException(500, f"TTS error chunk {i+1}: {r.text[:200]}")
                parts.append(r.content)

        audio_path = tmp_path / "audio.mp3"
        audio_path.write_bytes(b"".join(parts))

        # 2. Loudnorm
        norm_path = tmp_path / "audio_norm.mp3"
        rn = subprocess.run(
            ["ffmpeg", "-y", "-i", str(audio_path), "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
             "-c:a", "libmp3lame", "-q:a", "2", str(norm_path)],
            capture_output=True, text=True, timeout=120
        )
        if rn.returncode == 0:
            audio_path = norm_path

        # 3. Remove silence + adjust timestamps
        audio_path, silence_intervals = remove_silence(audio_path, tmp_path)

        # 4. Whisper
        from openai import OpenAI
        oai = OpenAI(api_key=oai_key)
        with open(audio_path, 'rb') as f:
            wresult = oai.audio.transcriptions.create(
                model="whisper-1", file=f,
                response_format="verbose_json",
                timestamp_granularities=["word"]
            )
        whisper_words = [{"word": w.word, "start": w.start, "end": w.end} for w in (wresult.words or [])]

        # 5. GPT subtitle split
        gpt_blocks = split_text_into_subtitle_blocks(wresult.text, oai)

        # 6. Align + adjust timestamps
        transcript_data = align_timestamps_python(gpt_blocks, whisper_words)
        transcript_data = adjust_timestamps(transcript_data, silence_intervals)
        transcript_data = split_long_blocks(transcript_data, font)

        # 7. Render
        bg_rgb   = hex_to_rgb(bg_color   if bg_color.startswith("#")   else f"#{bg_color}")
        text_rgb = hex_to_rgb(text_color if text_color.startswith("#") else f"#{text_color}")
        all_texts   = [b["text"] for b in transcript_data]
        global_size = fit_font_size(all_texts, font, font_size_int)
        pil_font    = load_font(font, global_size)

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
            render_frame(block["text"], bg_rgb, text_rgb, pil_font).save(str(fp), "PNG")
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
                "-c:v", "libx264", "-preset", "fast", "-profile:v", "high", "-level", "4.0",
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

        video_bytes = output.read_bytes()

    return Response(
        content=video_bytes,
        media_type="video/mp4",
        headers={"Content-Disposition": 'attachment; filename="video.mp4"'},
    )


@app.get("/", response_class=HTMLResponse)
def web_ui():
    return HTMLResponse(content=open(Path(__file__).parent / "index.html").read() if (Path(__file__).parent / "index.html").exists() else "<h1>Not found</h1>")


@app.get("/health")
def health():
    return {"status": "ok"}
