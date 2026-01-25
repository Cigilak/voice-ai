from fastapi import FastAPI, HTTPException
from pathlib import Path
import torch
import librosa
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    pipeline
)
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os

from fastapi import FastAPI
from config import Settings

# -----------------------------
# Authentication
# -----------------------------
security = HTTPBearer()

def authenticate(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    if credentials.credentials != Settings.api_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API token"
        )

Settings = Settings()
app = FastAPI(title = Settings.app_name)

@app.get("/health")
def health():
    return {
        "env": Settings.app_env,
        "db": Settings.database_url
    }
# -----------------------------
# Configuration
# -----------------------------
AUDIO_DIR = Path("audio_data")
SAMPLE_RATE = 16000

app = FastAPI(title="Speech & Sentiment API")

# -----------------------------
# Load models ONCE (startup)
# -----------------------------
processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base-960h"
)
asr_model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base-960h"
)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)


# -----------------------------
# Utility functions
# -----------------------------
def load_audio(file_path: Path):
    speech, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    return speech


def speech_to_text(audio):
    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        logits = asr_model(inputs.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    return transcription.lower()


def analyze_sentiment(text: str):
    return sentiment_pipeline(text)


# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/")
def root():
    return {"status": "API running"}


@app.get("/files", dependencies=[Depends(authenticate)])
def list_wav_files():
    if not AUDIO_DIR.exists():
        raise HTTPException(status_code=404, detail="Audio directory not found")

    files = [f.name for f in AUDIO_DIR.glob("*.wav")]
    return {"wav_files": files}


@app.get("/analyze/{filename}", dependencies=[Depends(authenticate)])
def analyze_audio(filename: str):
    file_path = AUDIO_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="WAV file not found")

    if file_path.suffix.lower() != ".wav":
        raise HTTPException(status_code=400, detail="Not a WAV file")

    audio = load_audio(file_path)
    transcription = speech_to_text(audio)
    sentiment = analyze_sentiment(transcription)

    return {
        "file": filename,
        "transcription": transcription,
        "sentiment": sentiment
    }
