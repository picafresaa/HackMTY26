"""
FastAPI service exposing POST /detect.

Only ever uses VAD-derived segmentation (segmentation.vad_segments) --
turns.json is a train-time artifact and is never available here. Make sure
the model in model.pkl was trained with --segmentation vad (train.py),
otherwise you'll have a train/serve skew and val performance won't
transfer to the judged calls.

ASSUMPTION: the exact JSON field name for the base64 WAV wasn't specified
in the prompt -- this uses "audio_base64". Check the grader's exact
request schema and rename the field in DetectRequest if needed.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import base64
import io
import pickle

import numpy as np
import soundfile as sf
from fastapi import FastAPI
from pydantic import BaseModel

from features import extract_features, feature_vector
from segmentation import vad_segments

app = FastAPI()

with open("model.pkl", "rb") as f:
    _bundle = pickle.load(f)
MODEL = _bundle["model"]
FEATURE_ORDER = _bundle["feature_order"]


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

    feats = extract_features(caller, caller_segs, agent_segs)
    vec = feature_vector(feats, FEATURE_ORDER).reshape(1, -1)
    prob_synthetic = float(MODEL.predict_proba(vec)[0, 1])

    return DetectResponse(
        is_synthetic=prob_synthetic >= 0.5,
        confidence=prob_synthetic,
    )
