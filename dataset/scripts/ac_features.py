"""Acustica global del caller (canal 0) -- fuente de informacion distinta a
tiempo (Capa 1) y a contenido de texto (Capa 2 / Plan B): mide como suena
la voz, no cuando ni cuanto habla. Nunca se proceso audio crudo antes de
esta sesion. Ver ARQUITECTURA.md seccion "Acustica global" y el plan de
esta sesion.

7 features congeladas ANTES de medir AUC (parametros de analisis tambien
fijados de antemano, valores estandar de voz, no ajustados a este dataset):
  f0_cv, jitter_local, shimmer_local, hnr_mean, spectral_flatness_mean,
  energy_cv, pause_floor_db (esta ultima con aviso: sospechosa de ser
  artefacto de pipeline, igual que caller_words).

Deliberadamente NO se mide velocidad de habla en esta ronda -- cualquier
proxy barato vuelve a depender de cuanto habla el caller (el confusor ya
detectado con caller_words).
"""
import csv
import os
import sys
import time
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l2_common import BASE, load_manifest, load_turns, merge_turns, layer1_score

AUDIO_DIR = os.path.join(BASE, "audio")

SR = 8000
FRAME_LEN = 0.032   # 32ms
HOP = 0.010         # 10ms
F0_MIN, F0_MAX = 75.0, 350.0
LAG_MIN = int(round(SR / F0_MAX))   # ~23
LAG_MAX = int(round(SR / F0_MIN))   # ~107
VOICED_THR = 0.35
EDGE_MARGIN = 0.05   # 50ms adentro de cada borde de turno
GAP_MIN = 0.3        # huecos >=300ms para pause_floor
GAP_TRIM = 0.1        # recorta 100ms de cada borde del hueco


def load_channel0(anon_id):
    path = os.path.join(AUDIO_DIR, anon_id + ".wav")
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 2 and w.getsampwidth() == 2
        sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype="<i2").reshape(-1, 2).astype(np.float64) / 32768.0
    return x[:, 0], sr


def caller_speech_regions(anon_id, margin=EDGE_MARGIN):
    """Turnos propios del caller (channel 0), recortados `margin` hacia
    adentro en cada borde para evitar artefactos de arranque/cierre."""
    turns = load_turns(anon_id)
    merged = merge_turns(turns)
    out = []
    for t in merged:
        if t["channel"] != 0:
            continue
        s, e = t["start"] + margin, t["end"] - margin
        if e - s > 0.05:
            out.append((s, e))
    return out


def caller_gap_regions(anon_id, min_gap=GAP_MIN, trim=GAP_TRIM):
    """Huecos donde el caller NO habla (entre sus propios turnos, y antes
    del primero / despues del ultimo dentro de la duracion de la llamada),
    recortando `trim` de cada borde. Se usa para pause_floor_db."""
    turns = load_turns(anon_id)
    merged = merge_turns(turns)
    caller_turns = sorted([(t["start"], t["end"]) for t in merged if t["channel"] == 0])
    if not caller_turns:
        return []
    call_end = max(t["end"] for t in merged)
    gaps = []
    prev_end = 0.0
    for s, e in caller_turns:
        if s - prev_end >= min_gap:
            gaps.append((prev_end, s))
        prev_end = max(prev_end, e)
    if call_end - prev_end >= min_gap:
        gaps.append((prev_end, call_end))
    out = []
    for s, e in gaps:
        s2, e2 = s + trim, e - trim
        if e2 - s2 > 0.05:
            out.append((s2, e2))
    return out


def frame_matrix(sig, sr, regions, frame_len=FRAME_LEN, hop=HOP):
    """Corta `sig` en frames de `frame_len` con salto `hop`, SOLO dentro de
    los intervalos de tiempo en `regions` (lista de (start,end) en seg).
    Devuelve (frames [n,L], run_id [n] -- entero que agrupa frames de un
    mismo tramo continuo, para no calcular jitter/shimmer cruzando huecos)."""
    L = int(round(frame_len * sr))
    H = int(round(hop * sr))
    frames, run_id = [], []
    for run, (s, e) in enumerate(regions):
        i0 = int(round(s * sr))
        i1 = int(round(e * sr))
        i = i0
        while i + L <= i1:
            frames.append(sig[i:i + L])
            run_id.append(run)
            i += H
    if not frames:
        return np.zeros((0, L)), np.zeros((0,), dtype=int)
    return np.array(frames), np.array(run_id)


def autocorr_pitch_hnr(frames):
    """Autocorrelacion normalizada por FFT, vectorizada sobre todos los
    frames a la vez. Devuelve para cada frame: (voiced_bool, period_samples,
    r_peak) usando el pico en el rango [LAG_MIN, LAG_MAX]."""
    n = frames.shape[0]
    if n == 0:
        return np.zeros(0, bool), np.zeros(0), np.zeros(0)
    L = frames.shape[1]
    x = frames - frames.mean(axis=1, keepdims=True)
    nfft = 1
    while nfft < 2 * L:
        nfft *= 2
    X = np.fft.rfft(x, n=nfft, axis=1)
    ac = np.fft.irfft(X * np.conj(X), n=nfft, axis=1)[:, :L].real
    r0 = ac[:, 0].copy()
    r0[r0 <= 1e-12] = 1e-12
    ac_norm = ac / r0[:, None]

    window = ac_norm[:, LAG_MIN:LAG_MAX + 1]
    best_lag = np.argmax(window, axis=1) + LAG_MIN
    r_peak = window[np.arange(n), np.argmax(window, axis=1)]
    voiced = r_peak >= VOICED_THR
    return voiced, best_lag.astype(float), r_peak


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


def compute_features(anon_id):
    sig, sr = load_channel0(anon_id)
    assert sr == SR, "unexpected sample rate " + str(sr) + " in " + anon_id

    speech_regions = caller_speech_regions(anon_id)
    frames, run_id = frame_matrix(sig, sr, speech_regions)
    n_speech_frames = frames.shape[0]

    out = {"n_speech_frames": n_speech_frames}
    if n_speech_frames < 10:
        for k in ("f0_cv", "jitter_local", "shimmer_local", "hnr_mean",
                   "spectral_flatness_mean", "energy_cv", "pause_floor_db"):
            out[k] = None
        return out

    voiced, period, r_peak = autocorr_pitch_hnr(frames)
    _, rms = db_energy(frames)
    flat = spectral_flatness(frames)

    out["spectral_flatness_mean"] = float(np.mean(flat))
    out["energy_cv"] = float(np.std(rms) / np.mean(rms)) if np.mean(rms) > 0 else None

    n_voiced = int(voiced.sum())
    if n_voiced >= 10:
        f0 = SR / period[voiced]
        out["f0_cv"] = float(np.std(f0) / np.mean(f0))
        r = np.clip(r_peak[voiced], 1e-4, 0.9999)
        out["hnr_mean"] = float(np.mean(10 * np.log10(r / (1 - r))))

        # jitter/shimmer: solo entre frames voiced-for-pitch consecutivos
        # (salto de 1 frame, HOP) DENTRO del mismo tramo continuo (run_id).
        idx = np.where(voiced)[0]
        jitters, shimmers = [], []
        for a, b in zip(idx[:-1], idx[1:]):
            if b - a == 1 and run_id[a] == run_id[b]:
                Ta, Tb = period[a], period[b]
                jitters.append(abs(Ta - Tb) / ((Ta + Tb) / 2.0))
                Aa, Ab = rms[a], rms[b]
                if (Aa + Ab) > 0:
                    shimmers.append(abs(Aa - Ab) / ((Aa + Ab) / 2.0))
        out["jitter_local"] = float(np.mean(jitters)) if len(jitters) >= 5 else None
        out["shimmer_local"] = float(np.mean(shimmers)) if len(shimmers) >= 5 else None
    else:
        out["f0_cv"] = None
        out["hnr_mean"] = None
        out["jitter_local"] = None
        out["shimmer_local"] = None

    gap_regions = caller_gap_regions(anon_id)
    gap_frames, _ = frame_matrix(sig, sr, gap_regions, frame_len=0.020, hop=0.020)
    if gap_frames.shape[0] >= 5:
        db, _ = db_energy(gap_frames)
        out["pause_floor_db"] = float(np.median(db))
    else:
        out["pause_floor_db"] = None

    return out


FEATURE_KEYS = ["f0_cv", "jitter_local", "shimmer_local", "hnr_mean",
                "spectral_flatness_mean", "energy_cv", "pause_floor_db"]


def audit_sample(rows, n=5):
    print("=== Auditoria (5 llamadas de train, a ojo) ===")
    sample = [r for r in rows if r["split"] == "train"][:n]
    for r in sample:
        t0 = time.time()
        f = compute_features(r["anon_id"])
        dt = time.time() - t0
        parts = []
        for k in FEATURE_KEYS:
            v = f[k]
            parts.append(k + "=" + ("%.3f" % v if v is not None else "None"))
        print(r["anon_id"] + " [" + r["label"] + "] n_speech_frames=" + str(f["n_speech_frames"]) +
              " t=" + str(round(dt * 1000)) + "ms  " + "  ".join(parts))
    print()


def main():
    manifest = load_manifest()
    audit_sample(manifest)

    t0 = time.time()
    rows = []
    l1 = {}
    for i, r in enumerate(manifest):
        anon_id = r["anon_id"]
        f = compute_features(anon_id)
        f["anon_id"] = anon_id
        f["label"] = r["label"]
        f["split"] = r["split"]
        l1[anon_id] = layer1_score(anon_id)
        rows.append(f)
        if (i + 1) % 50 == 0:
            print("  ... " + str(i + 1) + "/" + str(len(manifest)) +
                  "  (" + str(round(time.time() - t0, 1)) + "s transcurridos)")
    total_t = time.time() - t0
    print("\nTiempo total: " + str(round(total_t, 1)) + "s para " + str(len(manifest)) +
          " llamadas (" + str(round(total_t / len(manifest) * 1000)) + "ms/llamada)\n")

    f0_cvs = [r["f0_cv"] for r in rows if r["f0_cv"] is not None]
    print("Cobertura f0_cv: " + str(len(f0_cvs)) + "/" + str(len(rows)) +
          "  (CV medio=" + str(round(float(np.mean(f0_cvs)), 3)) + ", coef. de variacion, no Hz)\n")

    n_train_total = sum(1 for r in rows if r["split"] == "train")
    from l2_common import report_feature, spearman
    print("=== Acustica global: 7 features ===")
    results = {k: report_feature(rows, k, l1, n_train_total) for k in FEATURE_KEYS}

    from pb_fillers import build_dataset as build_filler_dataset
    filler_rows, _ = build_filler_dataset()
    cw = {r["anon_id"]: r["caller_words"] for r in filler_rows}
    print("\n=== rho contra caller_words (chequear que no se reintroduce el confusor) ===")
    for k in FEATURE_KEYS:
        pairs = [(r[k], cw.get(r["anon_id"])) for r in rows if r[k] is not None and cw.get(r["anon_id"]) is not None]
        rho = spearman([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else 0.0
        print("  " + k.ljust(24) + " rho_caller_words=" + ("%+.3f" % rho))

    passing = [k for k, v in results.items() if v and v["passes"]]
    print("\n=== Resumen: features que PASAN los 4 criterios: " + (str(passing) if passing else "(ninguna)") + " ===")

    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ac_features.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["anon_id", "label", "split", "n_speech_frames"] + FEATURE_KEYS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    print("\nGuardado: " + out_csv)


if __name__ == "__main__":
    main()
