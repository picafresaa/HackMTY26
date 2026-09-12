"""
Feature computation from already-segmented turns.

Segmentation is intentionally decoupled from feature computation (see
segmentation.py): at train time you can compute features from gold
turns/*.json (oracle, for diagnostics) or from your own VAD (production
parity); at serve time only VAD segments exist, since turns.json is never
sent to POST /detect. Always compare val performance using VAD-derived
features -- that's what the hidden set will actually give you.
"""

import numpy as np
import librosa

SAMPLE_RATE = 8000

# --- Turn-taking / interaction-dynamics features (primary signal) ---
# These are about *how the conversation happens*, not what the caller's
# voice sounds like -- they should generalize better to unseen voices,
# since train/val/hidden are speaker-disjoint.
TURN_TAKING_FEATURES = [
    "latency_mean", "latency_std", "latency_cv", "latency_p05",
    "latency_short_mean", "latency_long_mean", "latency_short_long_gap",
    "agent_bargein_count", "agent_bargein_persistence_mean",
    "caller_bargein_count", "caller_bargein_persistence_mean",
    "backchannel_rate", "n_caller_turns", "n_agent_turns", "speech_ratio",
]

# --- Caller acoustic-naturalness features (secondary signal) ---
# Useful, but risk overfitting to the specific TTS voices seen in train.
# A/B test with --drop-acoustic in train.py against val performance.
ACOUSTIC_FEATURES = [
    "pitch_jitter", "pitch_std", "spectral_flatness", "noise_floor",
]

FEATURE_ORDER = TURN_TAKING_FEATURES + ACOUSTIC_FEATURES


def turn_taking_features(caller_segs, agent_segs) -> dict:
    caller_segs = sorted(caller_segs)
    agent_segs = sorted(agent_segs)

    latencies, short_latencies, long_latencies = [], [], []
    for _, a_end in agent_segs:
        candidates = [c for c in caller_segs if c[0] >= a_end]
        if candidates:
            next_caller = min(candidates, key=lambda c: c[0])
            lat = next_caller[0] - a_end
            dur = next_caller[1] - next_caller[0]
            latencies.append(lat)
            (short_latencies if dur < 0.6 else long_latencies).append(lat)

    latencies = np.array(latencies) if latencies else np.array([0.0])
    short_latencies = np.array(short_latencies) if short_latencies else np.array([0.0])
    long_latencies = np.array(long_latencies) if long_latencies else np.array([0.0])

    # Agent talks over an in-progress caller turn -- does the caller keep
    # going regardless (pipeline unaware of being talked over) or cut off?
    agent_bargein_persistence = []
    for a_start, _ in agent_segs:
        for c_start, c_end in caller_segs:
            if c_start < a_start < c_end:
                agent_bargein_persistence.append(max(0.0, c_end - a_start))

    # Caller talks over an in-progress agent turn (natural human barge-in
    # behavior, or a bot's endpointing firing mid-agent-utterance).
    caller_bargein_persistence = []
    for c_start, _ in caller_segs:
        for a_start, a_end in agent_segs:
            if a_start < c_start < a_end:
                caller_bargein_persistence.append(max(0.0, a_end - c_start))

    caller_speech_time = sum(e - s for s, e in caller_segs)
    agent_speech_time = sum(e - s for s, e in agent_segs)
    total_time = max(caller_speech_time + agent_speech_time, 1e-6)

    short_segs = [s for s in caller_segs if (s[1] - s[0]) < 0.4]

    return {
        "latency_mean": float(np.mean(latencies)),
        "latency_std": float(np.std(latencies)),
        "latency_cv": float(np.std(latencies) / (np.mean(latencies) + 1e-6)),
        "latency_p05": float(np.percentile(latencies, 5)),
        "latency_short_mean": float(np.mean(short_latencies)),
        "latency_long_mean": float(np.mean(long_latencies)),
        "latency_short_long_gap": float(np.mean(long_latencies) - np.mean(short_latencies)),
        "agent_bargein_count": float(len(agent_bargein_persistence)),
        "agent_bargein_persistence_mean": float(np.mean(agent_bargein_persistence)) if agent_bargein_persistence else 0.0,
        "caller_bargein_count": float(len(caller_bargein_persistence)),
        "caller_bargein_persistence_mean": float(np.mean(caller_bargein_persistence)) if caller_bargein_persistence else 0.0,
        "backchannel_rate": len(short_segs) / max(len(agent_segs), 1),
        "n_caller_turns": float(len(caller_segs)),
        "n_agent_turns": float(len(agent_segs)),
        "speech_ratio": caller_speech_time / total_time,
    }


def caller_acoustic_features(caller: np.ndarray, caller_segs, sr: int = SAMPLE_RATE) -> dict:
    f0, voiced_flag, _ = librosa.pyin(
        caller.astype(float),
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
    )
    f0_voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])
    if len(f0_voiced) > 1:
        jitter_like = float(np.mean(np.abs(np.diff(f0_voiced))) / (np.mean(f0_voiced) + 1e-6))
        pitch_std = float(np.std(f0_voiced))
    else:
        jitter_like, pitch_std = 0.0, 0.0

    S = np.abs(librosa.stft(caller.astype(float), n_fft=512))
    spectral_flatness_mean = float(np.mean(librosa.feature.spectral_flatness(S=S)))

    # Noise floor: energy in the gaps between segments (real phone lines
    # are never truly silent; some TTS playback is closer to true zero).
    silence_energy = []
    prev_end = 0.0
    for start, end in sorted(caller_segs):
        if start > prev_end:
            chunk = caller[int(prev_end * sr):int(start * sr)]
            if len(chunk):
                silence_energy.append(float(np.mean(chunk.astype(float) ** 2)))
        prev_end = max(prev_end, end)
    noise_floor = float(np.mean(silence_energy)) if silence_energy else 0.0

    return {
        "pitch_jitter": jitter_like,
        "pitch_std": pitch_std,
        "spectral_flatness": spectral_flatness_mean,
        "noise_floor": noise_floor,
    }


def extract_features(caller: np.ndarray, caller_segs, agent_segs) -> dict:
    feats = {}
    feats.update(turn_taking_features(caller_segs, agent_segs))
    feats.update(caller_acoustic_features(caller, caller_segs))
    return feats


def feature_vector(feats: dict, feature_order=FEATURE_ORDER) -> np.ndarray:
    return np.array([feats[k] for k in feature_order], dtype=float)
