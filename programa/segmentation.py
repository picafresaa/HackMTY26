"""
Segmentation sources for turn-taking features.

- load_gold_turns(): reads turns/<anon_id>.json (train-time only -- this
  file is NEVER sent to POST /detect).
- vad_segments(): derives speech segments from raw audio via WebRTC VAD.
  This is the ONLY segmentation available in production, so features used
  by the served model must be computed from vad_segments(), not gold turns.

Use calibrate_vad.py to pick the aggressiveness value that best matches
the gold turns on the train split, then hardcode it here.
"""

import json
import numpy as np
import webrtcvad

SAMPLE_RATE = 8000
FRAME_MS = 30  # webrtcvad only supports 10/20/30 ms frames
FRAME_LEN = int(SAMPLE_RATE * FRAME_MS / 1000)

# Set this from calibrate_vad.py's output before training the production model.
DEFAULT_AGGRESSIVENESS = 2


def load_gold_turns(path):
    """turns/<anon_id>.json -> {0: [(start,end), ...], 1: [(start,end), ...]}"""
    with open(path) as f:
        data = json.load(f)
    segs = {0: [], 1: []}
    for t in data["turns"]:
        segs[t["channel"]].append((float(t["start"]), float(t["end"])))
    for ch in segs:
        segs[ch].sort()
    return segs


def _pcm16_frames(signal):
    n = len(signal) // FRAME_LEN
    signal = signal[: n * FRAME_LEN]
    pcm16 = (signal * 32767).astype(np.int16)
    for i in range(n):
        yield pcm16[i * FRAME_LEN:(i + 1) * FRAME_LEN].tobytes()


def vad_segments(signal, aggressiveness=DEFAULT_AGGRESSIVENESS):
    """Derive (start_s, end_s) speech segments from one raw audio channel."""
    vad = webrtcvad.Vad(aggressiveness)
    flags = []
    for frame_bytes in _pcm16_frames(signal):
        try:
            flags.append(vad.is_speech(frame_bytes, SAMPLE_RATE))
        except Exception:
            flags.append(False)
    flags = np.array(flags, dtype=bool)

    segments = []
    in_seg = False
    start = 0
    for i, flag in enumerate(flags):
        if flag and not in_seg:
            in_seg, start = True, i
        elif not flag and in_seg:
            in_seg = False
            segments.append((start * FRAME_MS / 1000, i * FRAME_MS / 1000))
    if in_seg:
        segments.append((start * FRAME_MS / 1000, len(flags) * FRAME_MS / 1000))
    return segments
