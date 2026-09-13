"""
FastAPI service exposing POST /detect.

Only ever uses VAD-derived segmentation (segmentation.vad_segments) --
turns.json is a train-time artifact and is never available here. Make sure
the model was trained with --segmentation vad (train.py) or, for
final_model.pkl, with combined_final.py's VAD-based pipeline -- otherwise
you'll have a train/serve skew and val performance won't transfer to the
judged calls.

Feature extraction is generic: it builds turn-taking features (features.py)
+ acoustic features (acoustic_v2.py) + optional text features (Scribe), and
only feeds the model the subset of keys named in its own feature_order. So
this same app.py works for model.pkl (train.py's model) or final_model.pkl
(the combined one) -- just point MODEL_PATH at whichever one you want to
serve.

ASSUMPTION: the exact JSON field name for the base64 WAV wasn't specified
in the prompt -- this uses "audio_base64". Check the grader's exact
request schema and rename the field in DetectRequest if needed.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000
    (or: set MODEL_PATH=final_model.pkl first, to serve the combined model)
"""

import base64
import io
import os
import pickle

import numpy as np
import soundfile as sf
from fastapi import FastAPI
from pydantic import BaseModel

from features import turn_taking_features, feature_vector
from segmentation import vad_segments
from acoustic_v2 import acoustic_features
from text_features import text_features, TEXT_FEATURE_ORDER

app = FastAPI()

MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
with open(MODEL_PATH, "rb") as f:
    _bundle = pickle.load(f)
MODEL = _bundle["model"]
FEATURE_ORDER = _bundle["feature_order"]

# Only import/call Scribe if the loaded model actually needs the text
# features -- avoids a live network call (and its cost/latency) on every
# request when the model doesn't use them.
NEEDS_TEXT_FEATURES = any(f in FEATURE_ORDER for f in TEXT_FEATURE_ORDER)
if NEEDS_TEXT_FEATURES:
    from transcribe import transcribe_channel


class DetectRequest(BaseModel):
    audio_base64: str


class DetectResponse(BaseModel):
    is_synthetic: bool
    confidence: float


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    raw = base64.b64decode(req.audio_base64)
    data, sr = sf.read(io.BytesIO(raw))

    if data.ndim == 1:
        caller = data
        agent = np.zeros_like(data)
    else:
        caller, agent = data[:, 0], data[:, 1]

    caller_segs = vad_segments(caller)
    agent_segs = vad_segments(agent)

    feats = {}
    feats.update(turn_taking_features(caller_segs, agent_segs))
    feats.update(acoustic_features(caller, caller_segs, sr=sr))

    if NEEDS_TEXT_FEATURES:
        # Adds one Scribe API call per request (network latency + cost).
        caller_transcript = transcribe_channel(caller)
        feats.update(text_features(caller_transcript))

    vec = feature_vector(feats, FEATURE_ORDER).reshape(1, -1)
    prob_synthetic = float(MODEL.predict_proba(vec)[0, 1])

    return DetectResponse(
        is_synthetic=prob_synthetic >= 0.5,
        confidence=prob_synthetic,
    )

