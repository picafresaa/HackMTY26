# Detector de caller humano vs. sintético (banco, español mexicano)

## Estrategia

Señal principal: **dinámica de turnos** (latencia de respuesta, quién
interrumpe a quién, si el caller sigue hablando cuando el agente le habla
encima). Esto explota el hecho de que te dan ambos canales y de que el
agente está diseñado para provocar reacciones (pide repetir info,
pregunta por cosas inexistentes, se calla, interrumpe). Un pipeline
autónomo (ASR + LLM + voz sintética) suele tener un **piso de latencia
casi constante** sin importar si la respuesta es "sí" o algo elaborado
(overhead del pipeline), y maneja mal las interrupciones.

Señal secundaria: naturalidad acústica del caller (jitter de pitch,
aplanamiento espectral, piso de ruido en silencios). Útil, pero con
riesgo de sobreajustar a las voces TTS específicas del train set, ya que
train/val/hidden son **speaker-disjoint** (voces nunca vistas). Por eso
`train.py` trae `--drop-acoustic` para comparar.

## ⚠️ El punto más importante: turns.json NO existe en producción

`POST /detect` solo recibe el WAV crudo. `turns/<anon_id>.json` es un
artefacto de entrenamiento. Si entrenas features con los turnos dorados
pero en el server usas tu propio VAD, hay desfase train/serve y el
accuracy de val no se va a sostener en la ronda evaluada.

Por eso:
1. `segmentation.py` separa `load_gold_turns()` (solo train, diagnóstico)
   de `vad_segments()` (el único disponible en `app.py`).
2. `calibrate_vad.py` mide qué tan bien tu VAD propio reproduce los
   turnos dorados, para elegir la agresividad que más se les parezca.
3. `train.py --segmentation vad` es el modelo que debes entrenar y
   servir. `--segmentation gold` solo te da un techo teórico (oráculo)
   para comparar, nunca lo despliegues.

## Flujo

```bash
pip install -r requirements.txt

# 1. calibra el VAD contra los turnos dorados
python calibrate_vad.py --manifest manifest.csv --audio-dir audio --turns-dir turns
# ajusta DEFAULT_AGGRESSIVENESS en segmentation.py con el mejor valor

# 2. entrena el modelo que realmente vas a servir
python train.py --manifest manifest.csv --audio-dir audio \
    --segmentation vad --out model.pkl

# (opcional) compara con el techo teórico usando turnos dorados
python train.py --manifest manifest.csv --audio-dir audio --turns-dir turns \
    --segmentation gold --out model_oracle.pkl

# (opcional) ¿las features acústicas ayudan o solo memorizan voces de train?
python train.py --manifest manifest.csv --audio-dir audio \
    --segmentation vad --drop-acoustic --out model_no_acoustic.pkl

# 3. levanta el servicio
uvicorn app:app --host 0.0.0.0 --port 8000
```

`train.py` imprime accuracy, AUC, y los pesos de cada feature en la
regresión -- revisa que las features acústicas no estén dominando el
modelo si el objetivo es generalizar a voces nunca vistas.

## Cosas a verificar contra el harness real

1. **Nombre exacto del campo JSON** del WAV en el POST -- asumí
   `audio_base64` en `DetectRequest` (`app.py`). Ajusta si el evaluador
   usa otro nombre.
2. **Calibración de `confidence`**: si premian calibración, considera
   Platt scaling / isotonic regression sobre el split val antes de la
   ronda automatizada.
3. Si tienes forma de transcribir el audio (ASR local), hay una capa de
   señal que este prototipo no usa todavía: si el caller repite mal la
   info que el agente le pide confirmar, o responde con seguridad a algo
   que el agente inventó (alucinación típica de LLM), eso es evidencia
   fuerte -- pero requiere ASR + análisis semántico, que no viene
   incluido aquí.
