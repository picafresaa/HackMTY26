"""
Verifica que exista un .wav en audio/ para cada anon_id del manifest.csv.

Uso:
    python check_audio.py
"""

import csv
import os

with open("manifest.csv", newline="") as f:
    rows = list(csv.DictReader(f))

missing = [r["anon_id"] for r in rows if not os.path.exists(f"audio/{r['anon_id']}.wav")]

print(f"faltan {len(missing)} de {len(rows)}")
if missing:
    print("ejemplos de los que faltan:", missing[:10])
