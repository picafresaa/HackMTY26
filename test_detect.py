"""
Prueba rápida de POST /detect sin depender de curl/base64 del sistema.
 
Uso:
    python test_detect.py audio/ALGUN_ID.wav
"""
 
import base64
import sys
import requests
 
if len(sys.argv) != 2:
    print("uso: python test_detect.py audio/ALGUN_ID.wav")
    sys.exit(1)
 
wav_path = sys.argv[1]
 
with open(wav_path, "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode("ascii")
 
resp = requests.post(
    "http://localhost:8000/detect",
    json={"audio_base64": audio_b64},
)
 
print("status:", resp.status_code)
print("respuesta:", resp.json())