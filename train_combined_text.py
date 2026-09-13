"""
Entrena el modelo combinado (turn-taking + acoustic_v2, las mismas 11
variables de final_model.pkl) SUMANDO las 3 features de texto de Scribe
(filler_rate, lexical_diversity, speaking_rate_cv), para ver si el texto
aporta algo mas alla de lo que ya capturan tiempos + acustica.

Requiere que ya hayas corrido build_transcripts.py (usa su cache en disco,
no vuelve a llamar a la API).

Uso:
    python train_combined_text.py --manifest manifest.csv --audio-dir audio \
        --transcripts-dir transcripts_text --out model_combined_text.pkl
"""

import argparse
import csv
import json
import os
import pickle

import numpy as np
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score

from features import turn_taking_features
from segmentation import vad_segments
from acoustic_v2 import acoustic_features
from text_features import text_features, TEXT_FEATURE_ORDER

# Las 11 ya validadas en final_model.pkl, mas las 3 de texto
FEATURE_ORDER = [
    "latency_cv", "latency_p05", "latency_long_mean", "latency_short_long_gap",
    "agent_bargein_count", "agent_bargein_persistence_mean",
    "caller_bargein_persistence_mean", "n_agent_turns", "speech_ratio",
    "energy_cv", "spectral_flatness_mean",
] + TEXT_FEATURE_ORDER


def load_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_transcript(transcripts_dir, anon_id):
    path = f"{transcripts_dir}/{anon_id}_caller.json"
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_dataset(rows, args):
    X, y = [], []
    n = len(rows)
    for i, row in enumerate(rows, start=1):
        anon_id = row["anon_id"]
        data, sr = sf.read(f"{args.audio_dir}/{anon_id}.wav")
        caller, agent = data[:, 0], data[:, 1]
        caller_segs, agent_segs = vad_segments(caller), vad_segments(agent)

        feats = {}
        feats.update(turn_taking_features(caller_segs, agent_segs))
        feats.update(acoustic_features(caller, caller_segs, sr=sr))

        transcript = load_transcript(args.transcripts_dir, anon_id)
        if transcript is not None:
            feats.update(text_features(transcript))
        else:
            feats.update({k: 0.0 for k in TEXT_FEATURE_ORDER})

        X.append([feats[k] for k in FEATURE_ORDER])
        y.append(1 if row["label"] == "synthetic" else 0)
        print(f"  [{i}/{n}] {anon_id}", flush=True)
    return np.array(X), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--transcripts-dir", required=True)
    ap.add_argument("--out", default="model_combined_text.pkl")
    args = ap.parse_args()

    rows = load_manifest(args.manifest)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]

    print("features:", FEATURE_ORDER)
    print(f"building features for {len(train_rows)} train / {len(val_rows)} val...")
    X_train, y_train = build_dataset(train_rows, args)
    X_val, y_val = build_dataset(val_rows, args)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    clf.fit(X_train, y_train)

    p_val = clf.predict_proba(X_val)[:, 1]
    pred_val = (p_val >= 0.5).astype(int)
    print("\nval accuracy:", accuracy_score(y_val, pred_val))
    print("val AUC:", roc_auc_score(y_val, p_val))

    coefs = clf.named_steps["logreg"].coef_[0]
    print("\npesos de cada feature (|coef| descendente):")
    for k, c in sorted(zip(FEATURE_ORDER, coefs), key=lambda kv: -abs(kv[1])):
        print(f"  {k:<24} {c:+.3f}")

    with open(args.out, "wb") as f:
        pickle.dump({"model": clf, "feature_order": FEATURE_ORDER}, f)
    print(f"\nmodelo guardado en {args.out}")

    print("\n=== Referencia (final_model.pkl, sin texto) ===")
    print("acc=0.958  AUC=0.967")


if __name__ == "__main__":
    main()