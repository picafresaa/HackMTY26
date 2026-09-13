"""
Wraps ElevenLabs Scribe v2 for channel-level transcription with word-level
timestamps. Used offline to build semantic/text features on top of the
turn-taking model. Could also be called from app.py at serving time now
that you've confirmed the judged environment has internet access -- just
know that adds one network round-trip (and ElevenLabs' per-call cost) to
every /detect request.

Requires:
    pip install elevenlabs
    ELEVENLABS_API_KEY environment variable set
"""

import io
import os

import numpy as np
import soundfile as sf
from elevenlabs import ElevenLabs

SAMPLE_RATE = 8000
_client = None


def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise RuntimeError("Set ELEVENLABS_API_KEY in your environment first.")
        _client = ElevenLabs(api_key=api_key)
    return _client


def _to_wav_buffer(signal: np.ndarray, sr: int = SAMPLE_RATE) -> io.BytesIO:
    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    buf.name = "channel.wav"  # SDK uses the name to infer the format
    return buf


def transcribe_channel(signal: np.ndarray, language_code: str = "spa") -> dict:
    """
    Transcribe one mono channel with Scribe v2.
    Returns {"text": str, "words": [{"word", "start", "end"}, ...]}.
    """
    client = get_client()
    wav_buf = _to_wav_buffer(signal)
    result = client.speech_to_text.convert(
        model_id="scribe_v2",
        file=wav_buf,
        language_code=language_code,
        tag_audio_events=True,
    )

    words = []
    for w in getattr(result, "words", None) or []:
        word = w.get("word") if isinstance(w, dict) else getattr(w, "word", None)
        start = w.get("start") if isinstance(w, dict) else getattr(w, "start", None)
        end = w.get("end") if isinstance(w, dict) else getattr(w, "end", None)
        if word is not None:
            words.append({"word": word, "start": start, "end": end})

    return {"text": getattr(result, "text", "") or "", "words": words}
