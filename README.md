# Voice Deepfake Detection

## HackMTY26 — Altur Challenge

An AI-powered system designed to detect synthetic voices in banking phone conversations.

## 📌 About the Challenge

For the HackMTY26 challenge, Altur asks teams to build a system capable of determining whether the incoming caller in a phone conversation is a real human or a synthetic voice.

Voice cloning technology has become increasingly realistic, creating new risks for banking phone calls. Attackers can impersonate customers or banks using synthetic voices, making voice alone less reliable as a method of authentication.

Our goal is to develop a prototype that can detect synthetic voices and help make phone-based banking more secure.

## 🎯 Objective

Build a system that takes a recorded phone conversation and determines whether the caller's voice is:

- 👤 Human
- 🤖 Synthetic

The system will expose an HTTP endpoint:

`POST /detect`

The endpoint will receive a stereo WAV recording encoded in base64.

- **Channel 0:** Caller
- **Channel 1:** AI agent
- **Sample rate:** 8 kHz

The response should contain a verdict such as:

```json
{
  "is_synthetic": true,
  "confidence": 0.87
}
