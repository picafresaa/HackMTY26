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
from pathlib import Path

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from features import turn_taking_features, feature_vector
from segmentation import vad_segments
from acoustic_v2 import acoustic_features
from text_features import text_features, TEXT_FEATURE_ORDER

load_dotenv()

app = FastAPI()

# Base model: turn-taking + acoustic only, no network call needed per
# request. This is what /detect uses unless the caller opts into the text
# model below. Defaults to the validated combined model (11 features from
# acoustic_v2.py) -- NOT model.pkl, which was trained against the older
# features.py acoustic features (pitch_jitter/noise_floor) and is
# incompatible with what this file actually computes below.
MODEL_PATH = os.environ.get("MODEL_PATH", "dataset/scripts/final_model.pkl")
with open(MODEL_PATH, "rb") as f:
    _bundle = pickle.load(f)
MODEL = _bundle["model"]
FEATURE_ORDER = _bundle["feature_order"]

# Optional text model: same base features + Scribe-derived text features.
# Requires an ElevenLabs API call per request, so it's opt-in via
# DetectRequest.use_text_model rather than always-on.
TEXT_MODEL_PATH = os.environ.get("TEXT_MODEL_PATH", "model_combined_text.pkl")
try:
    with open(TEXT_MODEL_PATH, "rb") as f:
        _text_bundle = pickle.load(f)
    TEXT_MODEL = _text_bundle["model"]
    TEXT_FEATURE_ORDER_FULL = _text_bundle["feature_order"]
except FileNotFoundError:
    TEXT_MODEL = None
    TEXT_FEATURE_ORDER_FULL = None

if TEXT_MODEL is not None:
    from transcribe import transcribe_channel


class DetectRequest(BaseModel):
    audio_base64: str
    use_text_model: bool = False


class DetectResponse(BaseModel):
    is_synthetic: bool
    confidence: float


@app.post("/detect", response_model=DetectResponse)
def detect(req: DetectRequest):
    if req.use_text_model and TEXT_MODEL is None:
        raise HTTPException(
            status_code=503,
            detail=f"Text model not available (missing {TEXT_MODEL_PATH}).",
        )

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

    if req.use_text_model:
        # Adds one Scribe API call per request (network latency + cost) --
        # only happens when the caller explicitly asks for it.
        try:
            caller_transcript = transcribe_channel(caller)
        except Exception as e:
            raise HTTPException(
                status_code=502, detail=f"ElevenLabs transcription failed: {e}"
            )
        feats.update(text_features(caller_transcript))
        model, feature_order = TEXT_MODEL, TEXT_FEATURE_ORDER_FULL
    else:
        model, feature_order = MODEL, FEATURE_ORDER

    vec = feature_vector(feats, feature_order).reshape(1, -1)
    prob_synthetic = float(model.predict_proba(vec)[0, 1])

    return DetectResponse(
        is_synthetic=prob_synthetic >= 0.5,
        confidence=prob_synthetic,
    )


@app.get("/")
def home():
    frontend_file = Path(__file__).resolve().parent / "frontend" / "index.html"
    return FileResponse(frontend_file)


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "text_model_loaded": TEXT_MODEL is not None,
    }

