"""
Demo del contrato HTTP de POST /detect (reto Altur / HackMTY26).

No depende de audios reales ni de librerías externas: genera un WAV
estereo 8kHz "de juguete" (silencio + un tono simple en cada canal)
solo para mostrar EXACTAMENTE la forma de la petición y la respuesta
esperada, tal como lo define app.py del repo NORA:

  POST /detect
  body: {"audio_base64": "<wav en base64>"}
  respuesta: {"is_synthetic": bool, "confidence": float}

Uso:
  python demo_detect_request.py                     # solo imprime la petición
  python demo_detect_request.py --send               # además intenta enviarla
  python demo_detect_request.py --send --url http://IP-DE-TU-COMPA:8000/detect
"""

import argparse
import base64
import io
import json
import math
import struct
import urllib.request
import wave

SAMPLE_RATE = 8000
DURATION_S = 1.0


def make_dummy_stereo_wav() -> bytes:
    """Genera un WAV estereo 8kHz de 1 segundo: canal 0 (caller) con un
    tono simple, canal 1 (agente) en silencio. Suficiente para probar
    la forma de la petición, NO para probar la calidad del modelo."""
    n_samples = int(SAMPLE_RATE * DURATION_S)
    frames = bytearray()
    for i in range(n_samples):
        t = i / SAMPLE_RATE
        caller = int(3000 * math.sin(2 * math.pi * 200 * t))  # tono 200Hz
        agent = 0  # silencio
        frames += struct.pack("<hh", caller, agent)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(bytes(frames))
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/detect",
                     help="endpoint a probar (default: %(default)s)")
    ap.add_argument("--send", action="store_true",
                     help="además de mostrar la petición, intenta enviarla")
    args = ap.parse_args()

    wav_bytes = make_dummy_stereo_wav()
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
    body = {"audio_base64": audio_b64}

    print("=== Petición HTTP que espera POST /detect ===")
    print(f"URL:     {args.url}")
    print("Método:  POST")
    print("Headers: Content-Type: application/json")
    print(f"Body (audio_base64 truncado): "
          f'{{"audio_base64": "{audio_b64[:40]}...{audio_b64[-10:]}"}}')
    print(f"(tamaño real del WAV: {len(wav_bytes)} bytes, "
          f"base64: {len(audio_b64)} chars)\n")

    print("Equivalente en curl:")
    print(f"  curl -X POST {args.url} \\")
    print("       -H 'Content-Type: application/json' \\")
    print("       -d '{\"audio_base64\": \"<pegar base64 aqui>\"}'\n")

    if not args.send:
        print("(usa --send para además intentar mandarla a un servidor real)")
        return

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        args.url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            print("=== Respuesta ===")
            print("status:", resp.status)
            print("body:  ", json.loads(resp.read().decode("utf-8")))
    except Exception as e:
        print("No se pudo conectar/enviar:", e)
        print("(revisa que --url apunte a un servidor con /detect corriendo)")


if __name__ == "__main__":
    main()