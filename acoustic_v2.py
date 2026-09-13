"""
Acoustic features from the teammate's ac_features.py / combined_final.py,
ported to be self-contained: no dependency on l2_common.py's hardcoded
folder-nesting assumptions, and computed ONLY on VAD-derived segments
(never on turns/*.json), since that's what's actually available in
production (POST /detect only gets raw audio).

Just the 2 features that made it into final_model.pkl: energy_cv and
spectral_flatness_mean.
"""

import numpy as np

EDGE_MARGIN = 0.05  # trim this much off each segment edge to dodge onset/offset artifacts


def frame_matrix(sig, sr, regions, frame_len=0.032, hop=0.010):
    """Cut `sig` into frame_len-second frames (hop between starts), only
    within the given (start_s, end_s) regions."""
    L = int(round(frame_len * sr))
    H = int(round(hop * sr))
    frames = []
    for s, e in regions:
        i0 = int(round(s * sr))
        i1 = int(round(e * sr))
        i = i0
        while i + L <= i1:
            frames.append(sig[i:i + L])
            i += H
    if not frames:
        return np.zeros((0, L))
    return np.array(frames)


def spectral_flatness(frames):
    if frames.shape[0] == 0:
        return np.zeros(0)
    win = np.hanning(frames.shape[1])
    x = frames * win[None, :]
    X = np.fft.rfft(x, axis=1)
    power = (np.abs(X) ** 2) + 1e-12
    gmean = np.exp(np.mean(np.log(power), axis=1))
    amean = np.mean(power, axis=1)
    return gmean / amean


def db_energy(frames):
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-10), rms


def acoustic_features(caller: np.ndarray, caller_segs, sr: int = 8000) -> dict:
    """energy_cv and spectral_flatness_mean over the caller's own VAD
    speech segments (edge-trimmed). Falls back to 0.0 when there's too
    little speech to measure anything reliably -- NOTE: the original
    validation script imputed missing values with the TRAIN set's median,
    which we don't have at serving time. 0.0 is a neutral placeholder,
    not a principled choice -- if this matters, store the train medians
    in the model bundle and impute with those instead.
    """
    regions = [
        (s + EDGE_MARGIN, e - EDGE_MARGIN)
        for s, e in caller_segs
        if e - s > 2 * EDGE_MARGIN + 0.05
    ]
    frames = frame_matrix(caller, sr, regions)
    if frames.shape[0] < 10:
        return {"energy_cv": 0.0, "spectral_flatness_mean": 0.0}

    flat = spectral_flatness(frames)
    _, rms = db_energy(frames)
    energy_cv = float(np.std(rms) / np.mean(rms)) if np.mean(rms) > 0 else 0.0
    return {"energy_cv": energy_cv, "spectral_flatness_mean": float(np.mean(flat))}
