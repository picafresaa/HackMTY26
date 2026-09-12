"""
Calibrate WebRTC VAD aggressiveness against the gold turns/*.json labels,
so the segmentation used in production (own VAD) tracks the gold turns
used to sanity-check training as closely as possible.

Usage:
    python calibrate_vad.py --manifest manifest.csv --audio-dir audio --turns-dir turns
"""

import argparse
import csv

import numpy as np
import soundfile as sf

from segmentation import vad_segments, load_gold_turns, FRAME_MS, SAMPLE_RATE


def segs_to_flags(segments, n_frames, frame_ms=FRAME_MS):
    flags = np.zeros(n_frames, dtype=bool)
    for start, end in segments:
        s = int(start * 1000 / frame_ms)
        e = int(end * 1000 / frame_ms)
        flags[s:min(e, n_frames)] = True
    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--turns-dir", required=True)
    ap.add_argument("--sample", type=int, default=50)
    args = ap.parse_args()

    with open(args.manifest, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "train"][: args.sample]

    for aggressiveness in [0, 1, 2, 3]:
        caller_accs, agent_accs = [], []
        for row in rows:
            anon_id = row["anon_id"]
            data, sr = sf.read(f"{args.audio_dir}/{anon_id}.wav")
            caller, agent = data[:, 0], data[:, 1]
            gold = load_gold_turns(f"{args.turns_dir}/{anon_id}.json")

            frame_len = int(SAMPLE_RATE * FRAME_MS / 1000)
            n_frames = len(caller) // frame_len

            gold_caller_flags = segs_to_flags(gold[0], n_frames)
            gold_agent_flags = segs_to_flags(gold[1], n_frames)

            pred_caller_flags = segs_to_flags(
                vad_segments(caller, aggressiveness=aggressiveness), n_frames
            )
            pred_agent_flags = segs_to_flags(
                vad_segments(agent, aggressiveness=aggressiveness), n_frames
            )

            caller_accs.append(np.mean(gold_caller_flags == pred_caller_flags))
            agent_accs.append(np.mean(gold_agent_flags == pred_agent_flags))

        print(
            f"aggressiveness={aggressiveness}: "
            f"caller frame-agreement={np.mean(caller_accs):.3f}, "
            f"agent frame-agreement={np.mean(agent_accs):.3f}"
        )

    print("\nPick the aggressiveness with the best combined agreement and set "
          "DEFAULT_AGGRESSIVENESS in segmentation.py to it.")


if __name__ == "__main__":
    main()
