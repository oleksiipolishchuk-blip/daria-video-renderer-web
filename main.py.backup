from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from typing import Optional
import base64
import subprocess
import tempfile
import os
import json
import sys
from pathlib import Path
import shutil

app = FastAPI()

VIDEO_WIDTH  = 720
VIDEO_HEIGHT = 1280

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
    max_width = int(VIDEO_WIDTH * 0.82)
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
    max_width = int(VIDEO_WIDTH * 0.82)
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


def render_frame(text: str, bg_rgb: tuple, text_rgb: tuple, font) -> "Image":
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (VIDEO_WIDTH, VIDEO_HEIGHT), bg_rgb)
    if not text.strip():
        return img
    draw = ImageDraw.Draw(img)

    lines = wrap_lines(draw, text, font, int(VIDEO_WIDTH * 0.82))
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


@app.get("/health")
def health():
    return {"status": "ok"}
