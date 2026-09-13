"""Modelo final oficial: VAD de Carlos (WebRTC calibrado) + 11 variables
(9 de conversacion de Carlos, tras confirmar por Lasso + reentrenamiento
que las otras 6 son redundantes; 2 de voz nuestras: energy_cv,
spectral_flatness_mean). Ver ARQUITECTURA.md, seccion "Modelo final
combinado", para la comparacion completa y el porque de cada decision.

Reemplaza a capa1_combined.py: aquel dependia de turns/<id>.json (VAD
oficial, que no existe en produccion). Este usa SOLO audio crudo de punta
a punta -- es el numero honesto: 95.8% acc en val, igual que el mejor
resultado que habiamos visto pero esta vez sin ayuda.

Guarda el modelo en dataset/scripts/final_model.pkl con el mismo formato
que los .pkl de Carlos ({"model", "feature_order"}) para que se pueda usar
directo en HackMTY26/app.py.

Uso:
  python final_model.py                  # entrena, reporta, guarda el .pkl
  from final_model import score_call      # para inferencia real desde WAV
"""
import os
import pickle
import sys

import numpy as np

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
from combined_final import build_dataset, to_matrix, report, our_acoustic_features, HACKMTY26

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

FEATURE_ORDER = [
    "latency_cv", "latency_p05", "latency_long_mean", "latency_short_long_gap",
    "agent_bargein_count", "agent_bargein_persistence_mean",
    "caller_bargein_persistence_mean", "n_agent_turns", "speech_ratio",
    "energy_cv", "spectral_flatness_mean",
]

MODEL_PATH = os.path.join(SCRIPTS_DIR, "final_model.pkl")


def extract_features(caller, agent, sr=8000):
    """Extraccion completa desde audio crudo -- para inferencia real
    (POST /detect). Usa el mismo VAD y las mismas funciones ya validadas
    en combined_final.py, solo recorta a las 11 variables finales."""
    sys.path.insert(0, HACKMTY26)
    from features import turn_taking_features
    from segmentation import vad_segments

    caller_segs, agent_segs = vad_segments(caller), vad_segments(agent)
    feats = turn_taking_features(caller_segs, agent_segs)
    feats.update(our_acoustic_features(caller, caller_segs, sr=sr))
    return {k: feats[k] for k in FEATURE_ORDER}


def score_features(model, feats):
    x = np.array([[feats[k] for k in model["feature_order"]]], dtype=float)
    return float(model["model"].predict_proba(x)[0, 1])


def score_wav(model, wav_path):
    import soundfile as sf
    sig, sr = sf.read(wav_path)
    caller, agent = sig[:, 0], sig[:, 1]
    feats = extract_features(caller, agent, sr=sr)
    return score_features(model, feats)


def load_model(path=MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    data = build_dataset(use_cache=True)
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

    print("=== Modelo final oficial: 11 variables, VAD real de punta a punta ===")
    report("umbral 0.5", yva, p_va, th=0.5)

    coefs = clf.named_steps["logreg"].coef_[0]
    print("\nPesos:")
    for k, c in sorted(zip(FEATURE_ORDER, coefs), key=lambda kv: -abs(kv[1])):
        print(f"  {k:<32} {c:+.3f}")

    print("\nErrores en val:")
    pred = (p_va >= 0.5).astype(int)
    for i in range(len(yva)):
        if pred[i] != yva[i]:
            print(f"  {ids_va[i]}  label={'synthetic' if yva[i]==1 else 'human'}  p={p_va[i]:.2f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": clf, "feature_order": FEATURE_ORDER}, f)
    print(f"\nModelo guardado: {MODEL_PATH}")

    print("\n=== Referencia ===")
    print("Modelo con 17 variables:              acc=0.958  AUC=0.967  TP=32 FP=1 TN=36 FN=2")
    print("Carlos, sin voz (su mejor version):    acc=0.944  AUC=0.966  TP=31 FP=1 TN=36 FN=3")


if __name__ == "__main__":
    main()
