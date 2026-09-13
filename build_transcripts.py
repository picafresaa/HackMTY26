"""
Batch-transcribes every call in the manifest with Scribe v2 and caches
results to <out-dir>/<anon_id>_caller.json / _agent.json. Run this ONCE --
train.py and any experimentation afterwards reads from the cache, so you
don't re-pay/re-call the API on every training run.

Cost estimate: Scribe v2 batch is ~$0.22/hour of audio. For ~350 calls
averaging ~2.5 min each (both channels), that's roughly $2-4 total.

Usage:
    python build_transcripts.py --manifest manifest.csv --audio-dir audio --out-dir transcripts_text
"""

import argparse
import csv
import json
import os
import time

import soundfile as sf
from dotenv import load_dotenv

from transcribe import transcribe_channel

load_dotenv()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out-dir", default="transcripts_text")
    ap.add_argument("--language-code", default="spa")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))

    t0 = time.time()
    for i, row in enumerate(rows, start=1):
        anon_id = row["anon_id"]
        caller_out = f"{args.out_dir}/{anon_id}_caller.json"
        agent_out = f"{args.out_dir}/{anon_id}_agent.json"

        if os.path.exists(caller_out) and os.path.exists(agent_out):
            print(f"  [{i}/{len(rows)}] {anon_id} (ya en caché, se salta)", flush=True)
            continue

        data, sr = sf.read(f"{args.audio_dir}/{anon_id}.wav")
        caller, agent = data[:, 0], data[:, 1]

        try:
            caller_t = transcribe_channel(caller, language_code=args.language_code)
            agent_t = transcribe_channel(agent, language_code=args.language_code)
        except Exception as e:
            print(f"  [{i}/{len(rows)}] {anon_id}  ERROR: {e}", flush=True)
            continue

        with open(caller_out, "w", encoding="utf-8") as f:
            json.dump(caller_t, f, ensure_ascii=False)
        with open(agent_out, "w", encoding="utf-8") as f:
            json.dump(agent_t, f, ensure_ascii=False)

        elapsed = time.time() - t0
        print(f"  [{i}/{len(rows)}] {anon_id}  ({elapsed/i:.1f}s/call avg)", flush=True)

    print("listo.")


if __name__ == "__main__":
    main()
