"""
Heuristic text-based features computed from Scribe transcripts. These
complement (not replace) the turn-taking/acoustic features in features.py.

These are cheap pattern-level signals, not semantic understanding:
- filler_rate: real callers use disfluencies ("eh", "este", "o sea"),
  scripted/TTS output usually doesn't unless written in on purpose.
- lexical_diversity: type-token ratio across the whole channel.
- speaking_rate_cv: how much the caller's words-per-second varies across
  the call -- humans speed up/slow down with content and emotion; many
  TTS pipelines are comparatively metronomic sentence to sentence.

NOT implemented here (stretch extension): actual semantic checks --
does the caller correctly repeat information the agent asked to confirm,
or confidently "confirm" something the agent invented. That needs an LLM
to judge each turn's content against the conversation so far, not just
word-level pattern stats.
"""

import re
import numpy as np

# Common Mexican Spanish disfluencies / filler words.
FILLERS = {
    "eh", "este", "esto", "pues", "mmm", "mm", "aja", "ajá", "osea",
    "digo", "bueno", "oye", "haber",
}


def _tokenize(text: str):
    return re.findall(r"[a-záéíóúñü]+", text.lower())


def filler_rate(text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    fillers = sum(1 for t in tokens if t in FILLERS)
    return fillers / len(tokens)


def lexical_diversity(text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def speaking_rate_variability(words: list) -> float:
    if len(words) < 10:
        return 0.0
    window = 5
    rates = []
    for i in range(0, len(words) - window, window):
        chunk = words[i:i + window]
        start, end = chunk[0].get("start"), chunk[-1].get("end")
        if start is None or end is None:
            continue
        dur = max(end - start, 1e-3)
        rates.append(window / dur)
    if len(rates) < 2:
        return 0.0
    return float(np.std(rates) / (np.mean(rates) + 1e-6))


def text_features(transcript: dict) -> dict:
    text = transcript.get("text", "") or ""
    words = transcript.get("words", []) or []
    return {
        "filler_rate": filler_rate(text),
        "lexical_diversity": lexical_diversity(text),
        "speaking_rate_cv": speaking_rate_variability(words),
    }


TEXT_FEATURE_ORDER = ["filler_rate", "lexical_diversity", "speaking_rate_cv"]
