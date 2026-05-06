import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

WIDTH = 1080
HEIGHT = 1920
FPS = 30

TEXT_W = 840
TEXT_X = (WIDTH - TEXT_W) // 2  # 120

SAFE_TOP = 270
SAFE_BOTTOM = 1440

OVERLAY_H    = int(HEIGHT * 0.47)
OVERLAY_FADE = 0.93

# Fonts are in the same directory as this file → /app/fonts/ on Railway
_FONT_BASE = Path(__file__).parent / "fonts"

# Curated font catalog with display names mapped to actual files
_NAMED_FONTS: list[dict] = [
    {"name": "Montserrat",  "file": "Montserrat-Bold.ttf"},
    {"name": "Gilroy",      "file": "Gilroy-Medium.ttf"},
    {"name": "Georgia",     "file": "Georgia.ttf"},
    {"name": "LT Carpet",   "file": "LTCarpet.ttf"},
    {"name": "Inter",       "file": "Inter-Medium.otf"},
    {"name": "Bodyhand",    "file": "Bodyhand Regular.otf"},
]

_FALLBACKS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
]


def _get_font(family: str, size: int) -> ImageFont.FreeTypeFont:
    # Named catalog lookup first
    for entry in _NAMED_FONTS:
        if entry["name"] == family:
            p = _FONT_BASE / entry["file"]
            if p.exists():
                return ImageFont.truetype(str(p), size)
    # Direct filename fallback (underscore-encoded family name)
    for ext in (".otf", ".ttf"):
        custom = _FONT_BASE / f"{family.replace(' ', '_')}{ext}"
        if custom.exists():
            return ImageFont.truetype(str(custom), size)
    for path in _FALLBACKS:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def list_fonts() -> list[dict]:
    """Return curated list of available fonts for the API."""
    return [
        {"name": entry["name"], "file": entry["file"]}
        for entry in _NAMED_FONTS
        if (_FONT_BASE / entry["file"]).exists()
    ]


def _wrap_lines(text: str, font: ImageFont.FreeTypeFont) -> list[str]:
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines = []
    for para in text.split("\n"):
        if not para.strip():
            lines.append("")
            continue
        words = para.split()
        cur = words[0]
        for w in words[1:]:
            test = cur + " " + w
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2] - bbox[0] <= TEXT_W:
                cur = test
            else:
                if w.startswith(("—", "–", "-")):
                    cur = test
                else:
                    lines.append(cur)
                    cur = w
        lines.append(cur)
    return lines


def _render_text(
    text: str,
    font: ImageFont.FreeTypeFont,
    color: str,
    align: Literal["left", "center", "right"] = "left",
    bottom_pad: bool = True,
) -> np.ndarray:
    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bbox = dummy.textbbox((0, 0), "Ay", font=font)
    line_h = int((bbox[3] - bbox[1]) * 1.5)

    lines = _wrap_lines(text, font)
    total_h = line_h * len(lines) + (line_h if bottom_pad else 0)

    img = Image.new("RGBA", (TEXT_W, total_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        if not line:
            continue
        lbbox = draw.textbbox((0, 0), line, font=font)
        line_w = lbbox[2] - lbbox[0]
        if align == "center":
            x = (TEXT_W - line_w) // 2
        elif align == "right":
            x = TEXT_W - line_w
        else:
            x = 0
        draw.text((x, i * line_h), line, font=font, fill=color)
    return np.array(img)


def _hex_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _is_light_bg(hex_color: str) -> bool:
    r, g, b = _hex_rgb(hex_color)
    return (0.299 * r + 0.587 * g + 0.114 * b) > 128


def _cover_crop(img: Image.Image, target_w: int, target_h: int, anchor_y: float = 0.5) -> Image.Image:
    orig_w, orig_h = img.size
    scale  = max(target_w / orig_w, target_h / orig_h)
    new_w  = int(orig_w * scale)
    new_h  = int(orig_h * scale)
    img    = img.resize((new_w, new_h), Image.LANCZOS)
    left   = (new_w - target_w) // 2
    top    = int((new_h - target_h) * float(np.clip(anchor_y, 0.0, 1.0)))
    return img.crop((left, top, left + target_w, top + target_h))


def _prepare_overlay(image_path: str, anchor_y: float = 0.5) -> Optional[np.ndarray]:
    try:
        raw = Image.open(image_path).convert("RGB")
        img = _cover_crop(raw, WIDTH, OVERLAY_H, anchor_y)
        img_blur = img.filter(ImageFilter.GaussianBlur(radius=1))

        sharp   = np.array(img).astype(np.float32)
        blurred = np.array(img_blur).astype(np.float32)

        fade_start = int(OVERLAY_H * OVERLAY_FADE)
        n_fade = OVERLAY_H - fade_start

        blend_w = np.zeros(OVERLAY_H, dtype=np.float32)
        blend_w[fade_start:] = np.linspace(0.0, 1.0, n_fade)
        bw = blend_w[:, np.newaxis, np.newaxis]
        rgb = (sharp * (1 - bw) + blurred * bw).astype(np.uint8)

        alpha = np.ones(OVERLAY_H, dtype=np.float32) * 255.0
        alpha[fade_start:] = np.linspace(255.0, 0.0, n_fade)

        result = np.zeros((OVERLAY_H, WIDTH, 4), dtype=np.uint8)
        result[:, :, :3] = rgb
        result[:, :, 3]  = np.broadcast_to(alpha[:, np.newaxis], (OVERLAY_H, WIDTH)).astype(np.uint8)
        return result
    except Exception as e:
        print(f"Overlay image error: {e}")
        return None


HOLD_SECS = 4.0
DISCLAIMER_BOTTOM_PAD = 380


def _audio_duration(path: str) -> float:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True,
        )
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def generate_video(
    text: str,
    title: str = "",
    title_font_family: str = "",
    title_font_size: int = 0,
    title_color: str = "",
    bg_color: str = "#0a0a0a",
    text_color: str = "#ffffff",
    font_family: str = "Arial Bold",
    font_size: int = 40,
    text_align: Literal["left", "center", "right"] = "left",
    audio_path: Optional[str] = None,
    music_path: Optional[str] = None,
    output_path: str = "output/video.mp4",
    scroll_speed: int = 150,
    bg_image_path: Optional[str] = None,
    bg_image_opacity: float = 0.5,
    overlay_image_path: Optional[str] = None,
    overlay_anchor_y: float = 0.5,
    disclaimer: str = "",
    on_progress: Optional[Callable[[int], None]] = None,
):
    font = _get_font(font_family, font_size)

    if title.strip():
        tff        = title_font_family if title_font_family else font_family
        tfs        = title_font_size   if title_font_size > 0 else int(font_size * 1.5)
        tc         = title_color       if title_color        else text_color
        title_font = _get_font(tff, tfs)
        title_rgba = _render_text(title, title_font, tc,         align=text_align, bottom_pad=False)
        body_rgba  = _render_text(text,  font,        text_color, align=text_align)
        combined_h = title_rgba.shape[0] + body_rgba.shape[0]
        combined   = np.zeros((combined_h, TEXT_W, 4), dtype=np.uint8)
        combined[:title_rgba.shape[0]]  = title_rgba
        combined[title_rgba.shape[0]:]  = body_rgba
        text_rgba  = combined
    else:
        text_rgba = _render_text(text, font, text_color, align=text_align)
    text_h, text_w = text_rgba.shape[:2]

    text_alpha = text_rgba[:, :, 3:4].astype(np.float32) / 255.0
    text_rgb   = text_rgba[:, :, :3].astype(np.float32)

    bg_solid = np.full((HEIGHT, WIDTH, 3), _hex_rgb(bg_color), dtype=np.float32)
    if bg_image_path and os.path.exists(bg_image_path):
        try:
            bg_img  = np.array(
                Image.open(bg_image_path).resize((WIDTH, HEIGHT), Image.LANCZOS).convert("RGB")
            ).astype(np.float32)
            op = float(np.clip(bg_image_opacity, 0.0, 1.0))
            bg_base = (bg_solid * (1.0 - op) + bg_img * op).astype(np.uint8)
        except Exception as e:
            print(f"BG image error: {e}")
            bg_base = bg_solid.astype(np.uint8)
    else:
        bg_base = bg_solid.astype(np.uint8)

    overlay_rgba = None
    if overlay_image_path and os.path.exists(overlay_image_path):
        overlay_rgba = _prepare_overlay(overlay_image_path, anchor_y=overlay_anchor_y)
    if overlay_rgba is not None:
        ov_alpha_pre = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0
        ov_rgb_pre   = overlay_rgba[:, :, :3].astype(np.float32)

    disc_alpha = disc_rgb = None
    disc_h = disc_w = 0
    disc_y = 0
    if disclaimer.strip():
        disc_color = "#555555" if _is_light_bg(bg_color) else "#ffffff"
        disc_font  = _get_font(font_family, 28)
        raw_disc   = _render_text(disclaimer.strip(), disc_font, disc_color, align="center", bottom_pad=False)
        raw_disc   = raw_disc.copy()
        raw_disc[:, :, 3] = (raw_disc[:, :, 3].astype(np.float32) * 0.8).astype(np.uint8)
        disc_h, disc_w = raw_disc.shape[:2]
        disc_alpha = raw_disc[:, :, 3:4].astype(np.float32) / 255.0
        disc_rgb   = raw_disc[:, :, :3].astype(np.float32)
        disc_y     = HEIGHT - disc_h - DISCLAIMER_BOTTOM_PAD

    y_start    = HEIGHT // 2
    y_end      = SAFE_BOTTOM - text_h
    total_dist = max(0, y_start - y_end)

    has_tts = bool(audio_path and os.path.exists(audio_path))
    has_music = bool(music_path and os.path.exists(music_path))

    if has_tts:
        scroll_dur = _audio_duration(audio_path)
        scroll_dur = scroll_dur if scroll_dur > 0 else 1.0
        px_per_sec = total_dist / scroll_dur if scroll_dur > 0 else 0
    else:
        px_per_sec = max(1, scroll_speed)
        scroll_dur = total_dist / px_per_sec if px_per_sec > 0 else 1.0

    duration = max(1.0, scroll_dur + HOLD_SECS)
    n_frames  = int(round(duration * FPS))

    tmp_raw = output_path + ".raw.mp4"

    frames_dir = Path(tempfile.mkdtemp(prefix="sv_frames_"))
    try:
        for fi in range(n_frames):
            t = fi / FPS
            y = int(y_start - min(t, scroll_dur) * px_per_sec)

            dst_y0 = max(0, y)
            dst_y1 = min(HEIGHT, y + text_h)
            frame  = bg_base.copy()

            if dst_y0 < dst_y1 and text_h > 0:
                src_y0 = dst_y0 - y
                src_y1 = dst_y1 - y
                alpha  = text_alpha[src_y0:src_y1]
                rgb    = text_rgb[src_y0:src_y1]
                region = frame[dst_y0:dst_y1, TEXT_X : TEXT_X + text_w].astype(np.float32)
                frame[dst_y0:dst_y1, TEXT_X : TEXT_X + text_w] = (
                    region * (1.0 - alpha) + rgb * alpha
                ).astype(np.uint8)

            if overlay_rgba is not None:
                region = frame[:OVERLAY_H].astype(np.float32)
                frame[:OVERLAY_H] = (
                    region * (1.0 - ov_alpha_pre) + ov_rgb_pre * ov_alpha_pre
                ).astype(np.uint8)

            if disc_alpha is not None and t > scroll_dur:
                fade   = min(1.0, (t - scroll_dur) / 0.5)
                dy0    = max(0, disc_y)
                dy1    = min(HEIGHT, disc_y + disc_h)
                if dy0 < dy1:
                    sy0    = dy0 - disc_y
                    sy1    = dy1 - disc_y
                    a_fade = disc_alpha[sy0:sy1] * fade
                    region = frame[dy0:dy1, TEXT_X:TEXT_X + disc_w].astype(np.float32)
                    frame[dy0:dy1, TEXT_X:TEXT_X + disc_w] = (
                        region * (1.0 - a_fade) + disc_rgb[sy0:sy1] * a_fade
                    ).astype(np.uint8)

            Image.fromarray(frame, "RGB").save(
                str(frames_dir / f"f{fi:06d}.jpg"), quality=85, optimize=False,
            )

        cmd_v = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-threads", "1",
            "-framerate", str(FPS),
            "-i", str(frames_dir / "f%06d.jpg"),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-threads", "1",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            tmp_raw,
        ]
        r = subprocess.run(cmd_v, capture_output=True, timeout=600)
        if r.returncode != 0:
            err = r.stderr.decode(errors="replace").strip()
            raise RuntimeError(f"FFmpeg encode failed (rc={r.returncode}): {err}")
    finally:
        shutil.rmtree(str(frames_dir), ignore_errors=True)

    # Mux audio if needed
    audio_inputs: list[str] = []
    if has_tts:
        audio_inputs.append(audio_path)
    if has_music:
        audio_inputs.append(music_path)

    if audio_inputs:
        cmd_a = ["ffmpeg", "-y", "-i", tmp_raw]
        for p in audio_inputs:
            cmd_a += ["-i", p]

        # Mix audio streams
        if len(audio_inputs) == 1:
            cmd_a += ["-map", "0:v", "-map", "1:a"]
            if has_music and not has_tts:
                cmd_a += ["-af", f"volume=1.0,atrim=duration={duration}"]
            cmd_a += ["-c:v", "copy", "-c:a", "aac", "-shortest"]
        else:
            # TTS at full vol, music at 0.1
            cmd_a += [
                "-filter_complex",
                f"[1:a]volume=1.0[tts];[2:a]volume=0.1,aloop=loop=-1:size=2e9,atrim=duration={duration}[mus];[tts][mus]amix=inputs=2:duration=first[aout]",
                "-map", "0:v", "-map", "[aout]",
                "-c:v", "copy", "-c:a", "aac",
            ]

        cmd_a.append(output_path)
        r = subprocess.run(cmd_a, capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"FFmpeg mux error: {r.stderr.decode(errors='replace')}")
        os.remove(tmp_raw)
    else:
        os.rename(tmp_raw, output_path)

    if on_progress:
        on_progress(100)
