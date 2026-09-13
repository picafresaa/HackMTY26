"""Version combinada: lo mejor de cada sistema, sin inventar nada nuevo.

  - Segmentacion:        WebRTC VAD calibrado de Carlos (HackMTY26/segmentation.py)
  - 15 features de conversacion: de Carlos (HackMTY26/features.py, TURN_TAKING_FEATURES)
  - 2 features de voz:   las 2 que mejor probamos nosotros (energy_cv,
                          spectral_flatness_mean, de ac_features.py), pero
                          calculadas sobre los segmentos de WebRTC VAD --
                          NO sobre turns/ oficiales, para que sea honesto
                          de punta a punta (production-matching) como la
                          version de Carlos.

Mismo pipeline y umbral que Carlos (StandardScaler + LogisticRegression
class_weight='balanced', umbral 0.5) para que la comparacion sea directa
contra sus dos numeros ya medidos:
  Carlos, sin voz (VAD real): 94.4% acc, AUC 0.966
  Carlos, con su voz (VAD real): 93.0% acc, AUC 0.983
"""
import csv
import os
import sys
import time

import numpy as np
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(SCRIPTS_DIR))
# Antes del merge del 13-sep, features.py/segmentation.py/manifest.csv
# vivian en la subcarpeta clonada HackMTY26/ -- ahora esa carpeta ES la
# raiz del repo (se fusiono con este folder), asi que ROOT == BASE.
ROOT = BASE
HACKMTY26 = ROOT  # alias, por si algo mas todavia lo importa con ese nombre
AUDIO = os.path.join(BASE, "audio")
MANIFEST = os.path.join(ROOT, "manifest.csv")

sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPTS_DIR)
from features import turn_taking_features, TURN_TAKING_FEATURES
from segmentation import vad_segments
from ac_features import frame_matrix, spectral_flatness, db_energy

FEATURE_ORDER = TURN_TAKING_FEATURES + ["energy_cv", "spectral_flatness_mean"]
EDGE_MARGIN = 0.05


def our_acoustic_features(caller, caller_segs, sr=8000):
    regions = [(s + EDGE_MARGIN, e - EDGE_MARGIN) for s, e in caller_segs if e - s > 2 * EDGE_MARGIN + 0.05]
    frames, _ = frame_matrix(caller, sr, regions)
    if frames.shape[0] < 10:
        return {"energy_cv": None, "spectral_flatness_mean": None}
    flat = spectral_flatness(frames)
    _, rms = db_energy(frames)
    energy_cv = float(np.std(rms) / np.mean(rms)) if np.mean(rms) > 0 else None
    return {"energy_cv": energy_cv, "spectral_flatness_mean": float(np.mean(flat))}


CACHE_CSV = os.path.join(SCRIPTS_DIR, "_combined_features.csv")


def build_dataset(use_cache=True):
    if use_cache and os.path.exists(CACHE_CSV):
        with open(CACHE_CSV, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        data = []
        for r in rows:
            d = {"anon_id": r["anon_id"], "label": r["label"], "split": r["split"]}
            for k in FEATURE_ORDER:
                d[k] = float(r[k]) if r[k] not in (None, "", "None") else None
            data.append(d)
        return data

    rows = list(csv.DictReader(open(MANIFEST, newline="", encoding="utf-8")))
    data = []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        sig, sr = sf.read(os.path.join(AUDIO, r["anon_id"] + ".wav"))
        caller, agent = sig[:, 0], sig[:, 1]
        caller_segs, agent_segs = vad_segments(caller), vad_segments(agent)

        feats = turn_taking_features(caller_segs, agent_segs)
        feats.update(our_acoustic_features(caller, caller_segs))
        feats["anon_id"] = r["anon_id"]
        feats["label"] = r["label"]
        feats["split"] = r["split"]
        data.append(feats)
        if i % 50 == 0:
            print(f"  [{i}/{len(rows)}]  {time.time()-t0:.0f}s")
    print(f"Total: {time.time()-t0:.0f}s para {len(rows)} llamadas ({(time.time()-t0)/len(rows)*1000:.0f}ms/llamada)\n")

    with open(CACHE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["anon_id", "label", "split"] + FEATURE_ORDER)
        w.writeheader()
        for d in data:
            w.writerow({k: d.get(k) for k in w.fieldnames})
    print(f"Cache guardado: {CACHE_CSV}\n")
    return data


def to_matrix(data, keys):
    import statistics as st
    train = [d for d in data if d["split"] == "train"]
    medians = {}
    for k in keys:
        vals = [d[k] for d in train if d[k] is not None]
        medians[k] = st.median(vals) if vals else 0.0
    X = np.zeros((len(data), len(keys)))
    for i, d in enumerate(data):
        for j, k in enumerate(keys):
            X[i, j] = d[k] if d[k] is not None else medians[k]
    y = np.array([1 if d["label"] == "synthetic" else 0 for d in data])
    split = np.array([d["split"] for d in data])
    ids = [d["anon_id"] for d in data]
    return X, y, split, ids


def report(name, y, p, th=0.5):
    pred = (p >= th).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    acc = (pred == y).mean()
    a = roc_auc_score(y, p)
    print(f"{name}: acc={acc:.3f}  AUC={a:.3f}  TP={tp} FP={fp} TN={tn} FN={fn}")
    return acc, a


def main():
    data = build_dataset()
    X, y, split, ids = to_matrix(data, FEATURE_ORDER)
    tr, va = split == "train", split == "val"
    Xtr, ytr, Xva, yva = X[tr], y[tr], X[va], y[va]
    ids_va = [ids[i] for i in range(len(ids)) if va[i]]

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    clf.fit(Xtr, ytr)
    p_va = clf.predict_proba(Xva)[:, 1]

    print("=== Combinado (VAD de Carlos + sus 15 features + nuestras 2 de voz) ===")
    report("umbral 0.5 (igual que Carlos)", yva, p_va, th=0.5)

    p_tr = clf.predict_proba(Xtr)[:, 1]
    cands = sorted(set(round(v, 3) for v in p_tr))
    best_th, best_acc = None, -1
    for th in cands:
        acc = ((p_tr >= th).astype(int) == ytr).mean()
        if acc > best_acc:
            best_acc, best_th = acc, th
    report(f"umbral {best_th:.3f} (elegido en train, nuestra disciplina)", yva, p_va, th=best_th)

    coefs = clf.named_steps["logreg"].coef_[0]
    print("\nPesos (|coef| descendente):")
    for k, c in sorted(zip(FEATURE_ORDER, coefs), key=lambda kv: -abs(kv[1])):
        print(f"  {k:<32} {c:+.3f}")

    print("\nErrores en val (umbral 0.5):")
    pred = (p_va >= 0.5).astype(int)
    for i in range(len(yva)):
        if pred[i] != yva[i]:
            print(f"  {ids_va[i]}  label={'synthetic' if yva[i]==1 else 'human'}  p={p_va[i]:.2f}")

    print("\n=== Referencia (todas medidas con VAD real, condicion de produccion) ===")
    print("Carlos sin voz:            acc=0.944  AUC=0.966  TP=31 FP=1 TN=36 FN=3")
    print("Carlos con su voz:         acc=0.930  AUC=0.983  TP=31 FP=2 TN=35 FN=3")
    print("Nuestro combinado (VAD propio, simple): acc=0.859  AUC=0.936  TP=31 FP=7 TN=30 FN=3")


if __name__ == "__main__":
    main()
