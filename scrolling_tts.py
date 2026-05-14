import os
from elevenlabs.client import ElevenLabs

CHUNK_LIMIT = 4900


def get_sv_voices() -> list[dict]:
    try:
        client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY", ""))
        voices = client.voices.get_all().voices
        return [{"id": v.voice_id, "name": v.name} for v in voices]
    except Exception as e:
        print(f"Failed to fetch voices: {e}")
        return []


def _split_sentences(text: str) -> list[str]:
    if len(text) <= CHUNK_LIMIT:
        return [text]
    chunks = []
    remaining = text
    while len(remaining) > CHUNK_LIMIT:
        cut = remaining.rfind(".", 0, CHUNK_LIMIT)
        if cut == -1:
            cut = remaining.rfind(" ", 0, CHUNK_LIMIT)
        if cut == -1:
            cut = CHUNK_LIMIT - 1
        cut += 1
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _generate_chunk(text: str, voice_id: str, output_path: str) -> bool:
    try:
        client = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY", ""))
        audio = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id="eleven_v3",
            voice_settings={"stability": 0.5, "similarity_boost": 0.75},
        )
        with open(output_path, "wb") as f:
            for chunk in audio:
                f.write(chunk)
        return True
    except Exception as e:
        print(f"TTS chunk failed: {e}")
        return False


def generate_sv_tts(text: str, voice_id: str, output_path: str) -> bool:
    chunks = _split_sentences(text)

    if len(chunks) == 1:
        return _generate_chunk(text, voice_id, output_path)

    from moviepy.editor import AudioFileClip, concatenate_audioclips

    part_paths = [output_path.replace(".mp3", f"_part{i}.mp3") for i in range(len(chunks))]
    try:
        for chunk, path in zip(chunks, part_paths):
            if not _generate_chunk(chunk, voice_id, path):
                return False
        clips = [AudioFileClip(p) for p in part_paths]
        final = concatenate_audioclips(clips)
        final.write_audiofile(output_path, logger=None)
        for c in clips:
            c.close()
        final.close()
        return True
    except Exception as e:
        print(f"TTS concatenation failed: {e}")
        return False
    finally:
        for p in part_paths:
            if os.path.exists(p):
                os.remove(p)
