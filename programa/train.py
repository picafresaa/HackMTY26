"""
Train a human-vs-synthetic classifier from a manifest CSV.

Manifest format (one row per call):
    anon_id,label,split,duration_s

Files expected:
    <audio_dir>/{anon_id}.wav   stereo 8kHz, ch0=caller, ch1=agent
    <turns_dir>/{anon_id}.json  gold turns (optional, train-time only)

IMPORTANT: gold turns/*.json are NOT available at serving time (POST
/detect only gets raw audio). Use --segmentation vad to train the model
you'll actually ship -- that's what matches production. Use --segmentation
gold only to get an oracle/upper-bound number for comparison, never ship
that model.

Usage:
    python train.py --manifest manifest.csv --audio-dir audio \
        --segmentation vad --out model.pkl
"""

import argparse
import csv
import pickle

import numpy as np
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score

from features import extract_features, feature_vector, FEATURE_ORDER, TURN_TAKING_FEATURES
from segmentation import load_gold_turns, vad_segments


def load_manifest(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def get_segments(anon_id, caller, agent, args):
    if args.segmentation == "gold":
        gold = load_gold_turns(f"{args.turns_dir}/{anon_id}.json")
        return gold[0], gold[1]
    return vad_segments(caller), vad_segments(agent)


def build_dataset(manifest_rows, args, feature_order):
    X, y = [], []
    for row in manifest_rows:
        anon_id = row["anon_id"]
        data, sr = sf.read(f"{args.audio_dir}/{anon_id}.wav")
        assert data.ndim == 2 and data.shape[1] == 2, f"expected stereo wav for {anon_id}"
        caller, agent = data[:, 0], data[:, 1]

        caller_segs, agent_segs = get_segments(anon_id, caller, agent, args)
        feats = extract_features(caller, caller_segs, agent_segs)
        X.append(feature_vector(feats, feature_order))
        y.append(1 if row["label"] == "synthetic" else 0)
    return np.array(X), np.array(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--turns-dir", default="turns",
                     help="only needed if --segmentation gold")
    ap.add_argument("--segmentation", choices=["vad", "gold"], default="vad",
                     help="vad = production-matching (ship this); "
                          "gold = oracle upper bound, for comparison only")
    ap.add_argument("--drop-acoustic", action="store_true",
                     help="train on turn-taking features only, to check "
                          "how much the acoustic features are helping vs. "
                          "just overfitting to train-set voices")
    ap.add_argument("--out", default="model.pkl")
    args = ap.parse_args()

    feature_order = TURN_TAKING_FEATURES if args.drop_acoustic else FEATURE_ORDER

    rows = load_manifest(args.manifest)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]

    print(f"segmentation={args.segmentation}, features={len(feature_order)} "
          f"({'turn-taking only' if args.drop_acoustic else 'turn-taking + acoustic'})")
    print(f"building features for {len(train_rows)} train / {len(val_rows)} val calls...")

    X_train, y_train = build_dataset(train_rows, args, feature_order)
    X_val, y_val = build_dataset(val_rows, args, feature_order)

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    clf.fit(X_train, y_train)

    val_probs = clf.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.5).astype(int)
    print("val accuracy:", accuracy_score(y_val, val_preds))
    print("val AUC:", roc_auc_score(y_val, val_probs))

    coefs = clf.named_steps["logreg"].coef_[0]
    print("\nfeature weights (|coef| descending):")
    for name, c in sorted(zip(feature_order, coefs), key=lambda x: -abs(x[1])):
        print(f"  {name:32s} {c:+.3f}")

    with open(args.out, "wb") as f:
        pickle.dump({"model": clf, "feature_order": feature_order}, f)
    print(f"\nmodel saved to {args.out}")


if __name__ == "__main__":
    main()
