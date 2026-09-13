"""Utilidades compartidas para validar la Capa 2 (conductual) con las 353
llamadas transcritas. Sin sklearn/scipy en el entorno -> AUC por rangos y
regresion logistica hechos a mano con numpy, igual que en probe_turns.py /
probe_vad.py.
"""
import csv
import json
import os
import re
import unicodedata
import statistics as st

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # .../hackaton2026
MANIFEST = os.path.join(BASE, "dataset", "manifest.csv")
TURNS_DIR = os.path.join(BASE, "dataset", "turns")
TRANSCRIPTS_DIR = os.path.join(BASE, "dataset", "transcripts")


def load_manifest():
    return list(csv.DictReader(open(MANIFEST, encoding="utf-8")))


def load_turns(anon_id):
    p = os.path.join(TURNS_DIR, anon_id + ".json")
    return json.load(open(p, encoding="utf-8"))["turns"]


def load_transcript(anon_id):
    p = os.path.join(TRANSCRIPTS_DIR, anon_id + ".json")
    return json.load(open(p, encoding="utf-8"))


def norm(text):
    """minusculas, sin acentos, sin puntuacion (para regex de contenido)."""
    t = unicodedata.normalize("NFKD", text.lower())
    t = t.encode("ascii", "ignore").decode()
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def merge_turns(turns, gap=0.6):
    """Misma regla que probe2.py / probe_vad.py: junta segmentos del mismo
    canal separados por menos de `gap` segundos en una sola 'intervencion'."""
    turns = sorted(turns, key=lambda t: t["start"])
    out = []
    for t in turns:
        if out and out[-1]["channel"] == t["channel"] and t["start"] - out[-1]["end"] < gap:
            out[-1]["end"] = max(out[-1]["end"], t["end"])
        else:
            out.append(dict(t))
    return out


def response_latencies(merged):
    """Lista de latencias agente-calla -> caller-habla, en orden, con el
    indice del turno de agente que las origina (dentro de `merged`)."""
    out = []
    for i, t in enumerate(merged):
        if t["channel"] == 0:
            prev_idx = None
            for j in range(i - 1, -1, -1):
                if merged[j]["channel"] == 1:
                    prev_idx = j
                    break
            if prev_idx is not None:
                gap = t["start"] - merged[prev_idx]["end"]
                if -1.0 < gap < 8.0:
                    out.append((gap, prev_idx, i))
    return out


def layer1_score(anon_id):
    """Latencia mediana de respuesta (Capa 1), igual definicion que
    probe2.py, para poder reproducir AUC 0.981 / val 94.4% como sanity."""
    turns = load_turns(anon_id)
    merged = merge_turns(turns)
    lat = [g for g, _, _ in response_latencies(merged)]
    return st.median(lat) if lat else None


def auc(pos, neg):
    """AUC por rangos (Mann-Whitney U), pos = valores de la clase synthetic."""
    pos = [v for v in pos if v is not None]
    neg = [v for v in neg if v is not None]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))

def auc_labeled(vals, labels):
    pos = [v for v, l in zip(vals, labels) if l == "synthetic"]
    neg = [v for v, l in zip(vals, labels) if l == "human"]
    return auc(pos, neg)


def bootstrap_ci(vals, labels, n=2000, seed=0):
    """IC95% de AUC por bootstrap de pares (resamplea filas, no clases)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(vals))
    vals = np.array(vals, dtype=object)
    labels = np.array(labels, dtype=object)
    boots = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        a = auc_labeled(vals[s].tolist(), labels[s].tolist())
        if a is not None:
            boots.append(a)
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots)) - 1]
    return lo, hi


def fit_threshold(vals, labels):
    """Mejor umbral (val >= th => synthetic) ajustado SOLO sobre lo que se
    le pase (debe ser train)."""
    cands = sorted(set(round(v, 3) for v in vals if v is not None))
    best = None
    for th in cands:
        acc = sum((v >= th) == (l == "synthetic") for v, l in zip(vals, labels) if v is not None) / len(vals)
        if best is None or acc > best[1]:
            best = (th, acc)
    return best


def eval_threshold(th, vals, labels):
    tp = fp = tn = fn = 0
    for v, l in zip(vals, labels):
        if v is None:
            continue
        pred = v >= th
        truth = l == "synthetic"
        if pred and truth: tp += 1
        elif pred and not truth: fp += 1
        elif not pred and not truth: tn += 1
        else: fn += 1
    n = tp + fp + tn + fn
    acc = (tp + tn) / n if n else None
    return {"acc": acc, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": n}


def spearman(a, b):
    """Correlacion de Spearman sin scipy: Pearson sobre rangos."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def logreg_fit(X, y, l2=1.0, lr=0.1, iters=3000):
    """Regresion logistica simple por descenso de gradiente con L2, en numpy
    puro (no hay sklearn en el entorno). X: (n,d) ya estandarizado; y: 0/1."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    w = np.zeros(d + 1)
    for _ in range(iters):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-z))
        grad = Xb.T @ (p - y) / n
        grad[1:] += l2 * w[1:] / n
        w -= lr * grad
    return w


def logreg_predict(w, X):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    Xb = np.hstack([np.ones((n, 1)), X])
    z = Xb @ w
    return 1.0 / (1.0 + np.exp(-z))


def report_feature(rows, key, l1_map, n_train_total=None, band_lo=1.4, band_hi=2.4):
    """Evalua una feature contra los 4 criterios fijados en el plan de
    validacion de Capa 2: AUC train >=0.70 con IC95%>0.5, AUC val >=0.65,
    cobertura >=70% de train, y AUC >=0.60 DENTRO de la banda ambigua de
    Capa 1 (band_lo-band_hi) -- que es el unico lugar donde a una feature
    de Capa 2 le toca aportar en la cascada. rows: lista de dicts con al
    menos {"anon_id","label","split",key}. l1_map: anon_id -> layer1_score.
    Imprime una linea de reporte y devuelve el resultado o None si la
    cobertura es insuficiente para medir nada."""
    def in_band(anon_id):
        l1v = l1_map.get(anon_id)
        return l1v is not None and band_lo <= l1v <= band_hi

    train = [(r[key], r["label"]) for r in rows if r["split"] == "train" and r[key] is not None]
    val = [(r[key], r["label"]) for r in rows if r["split"] == "val" and r[key] is not None]
    if len(train) < 10 or len(val) < 5:
        print(f"  {key:<24} cobertura insuficiente (train n={len(train)}, val n={len(val)}) -- SKIP")
        return None

    tr_vals, tr_labs = zip(*train)
    va_vals, va_labs = zip(*val)
    a_tr = auc_labeled(list(tr_vals), list(tr_labs))
    lo, hi = bootstrap_ci(list(tr_vals), list(tr_labs))
    a_va = auc_labeled(list(va_vals), list(va_labs))

    band_train = [(r[key], r["label"]) for r in rows if r["split"] == "train" and r[key] is not None and in_band(r["anon_id"])]
    band_val = [(r[key], r["label"]) for r in rows if r["split"] == "val" and r[key] is not None and in_band(r["anon_id"])]
    a_band_tr = auc_labeled([v for v, l in band_train], [l for v, l in band_train]) if len(band_train) >= 8 else None
    a_band_va = auc_labeled([v for v, l in band_val], [l for v, l in band_val]) if len(band_val) >= 4 else None

    pairs = [(r[key], l1_map[r["anon_id"]]) for r in rows if r[key] is not None and l1_map.get(r["anon_id"]) is not None]
    rho = spearman([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else 0.0

    # Simetrico: una feature puede separar fuerte "al reves" (AUC << 0.5,
    # ej. bargein_rate mas alto en human) y eso es tan buena señal como
    # AUC >> 0.5 -- lo unico que importa es |AUC-0.5|, no el signo. Los
    # umbrales del plan (0.70 / 0.65 / 0.60) se aplican a la version
    # "efectiva" (max(a, 1-a)); direction dice de que lado quedo.
    eff_tr = max(a_tr, 1 - a_tr) if a_tr is not None else None
    eff_va = max(a_va, 1 - a_va) if a_va is not None else None
    eff_band_tr = max(a_band_tr, 1 - a_band_tr) if a_band_tr is not None else None
    ci_away_from_chance = (lo > 0.5) or (hi < 0.5)
    direction = "synth>human" if (a_tr is not None and a_tr >= 0.5) else "human>synth"

    raw_bar = (eff_tr is not None and eff_tr >= 0.70 and ci_away_from_chance
               and eff_va is not None and eff_va >= 0.65)
    coverage_ok = (len(train) / n_train_total) >= 0.70 if n_train_total else True
    band_ok = eff_band_tr is not None and eff_band_tr >= 0.60
    passes = raw_bar and coverage_ok and band_ok

    if a_band_tr is not None and a_band_va is not None:
        band_str = f"{a_band_tr:.3f}/{a_band_va:.3f} (n={len(band_train)}/{len(band_val)})"
    else:
        band_str = "n/a (pocos casos en banda)"
    tag = "PASA" if passes else ("no pasa -- falla banda ambigua" if raw_bar and not band_ok else
                                  ("no pasa -- falla cobertura" if raw_bar and not coverage_ok else "no pasa"))
    print(f"  {key:<24} n_tr={len(train):<4} n_val={len(val):<4} "
          f"AUC_tr={a_tr:.3f} [{lo:.3f},{hi:.3f}]  AUC_val={a_va:.3f}  "
          f"AUC_banda(tr/val)={band_str}  rho_L1={rho:+.3f}  dir={direction}  {tag}")
    return {"auc_tr": a_tr, "auc_val": a_va, "passes": passes, "n_train": len(train), "n_val": len(val)}


def standardize(X_train, X_other=()):
    X_train = np.asarray(X_train, dtype=float)
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1.0
    out = [(X_train - mu) / sd]
    for X in X_other:
        out.append((np.asarray(X, dtype=float) - mu) / sd)
    return out if len(out) > 1 else out[0]
