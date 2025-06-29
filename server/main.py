import base64
import io
import os
import tempfile
import wave
import torch
import numpy as np
import librosa
from typing import List
from pydantic import BaseModel

from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse

from TTS.api import TTS
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager

torch.set_num_threads(int(os.environ.get("NUM_THREADS", os.cpu_count())))
device = torch.device("cuda" if os.environ.get("USE_CPU", "0") == "0" else "cpu")

if not torch.cuda.is_available() and device == "cuda":
    raise RuntimeError("CUDA device unavailable, please use Dockerfile.cpu instead.")

custom_model_path = os.environ.get("CUSTOM_MODEL_PATH", "/app/tts_models")
default_model_name = os.environ.get(
    "DEFAULT_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2"
)

print(f"Loading model: {default_model_name}", flush=True)
model = TTS(
    model_name=default_model_name, progress_bar=False, gpu=device.type == "cuda"
)

# Determine model type after loading
MODEL_TYPE = "xtts" if "xtts" in default_model_name.lower() else "vits"
print(f"Running {MODEL_TYPE.upper()} Server ...", flush=True)

##### Run fastapi #####
app = FastAPI(
    title="XTTS Streaming server",
    description="""XTTS Streaming server""",
    version="0.0.1",
    docs_url="/",
)


def postprocess(wav):
    """Post process the output waveform"""
    if isinstance(wav, list):
        wav = torch.cat(wav, dim=0)
    wav = wav.clone().detach().cpu().numpy()
    wav = wav[None, : int(wav.shape[0])]
    wav = np.clip(wav, -1, 1)
    wav = (wav * 32767).astype(np.int16)
    return wav


def encode_audio_common(
    frame_input, encode_base64=True, sample_rate=24000, sample_width=2, channels=1
):
    """Return base64 encoded audio"""
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    if encode_base64:
        b64_encoded = base64.b64encode(wav_buf.getbuffer()).decode("utf-8")
        return b64_encoded
    else:
        return wav_buf.read()


class StreamingInputs(BaseModel):
    text: str
    language: str = None
    speaker_embedding: List[float] = None
    gpt_cond_latent: List[List[float]] = None
    speaker_idx: str = None  # For VITS models
    speed: float = 1.0  # Speed control: 0.5 = slower, 2.0 = faster
    add_wav_header: bool = True
    stream_chunk_size: int = 20


@app.post("/tts_stream")
def predict_streaming_endpoint(parsed_input: StreamingInputs):
    """Stream speech generation from text"""

    def stream():
        try:
            # Prepare TTS arguments
            tts_kwargs = {"text": parsed_input.text}

            # Add speaker if specified
            if parsed_input.speaker_idx is not None:
                tts_kwargs["speaker"] = parsed_input.speaker_idx

            # Add language if specified (mainly for XTTS)
            if parsed_input.language is not None:
                tts_kwargs["language"] = parsed_input.language

            # Add speed if specified
            if parsed_input.speed is not None:
                tts_kwargs["speed"] = parsed_input.speed

            # Generate speech
            out = model.tts(**tts_kwargs)

            # Convert to audio format
            wav = np.array(out)
            wav = np.clip(wav, -1, 1)
            wav = (wav * 32767).astype(np.int16)

            # Stream in chunks
            chunk_size = parsed_input.stream_chunk_size * 1024
            if parsed_input.add_wav_header:
                yield encode_audio_common(b"", encode_base64=False)

            for i in range(0, len(wav), chunk_size):
                yield wav[i : i + chunk_size].tobytes()

        except Exception as e:
            # For streaming, we can't raise HTTP exceptions, so we'll just stop the stream
            print(f"Streaming TTS generation failed: {str(e)}", flush=True)
            return

    return StreamingResponse(stream(), media_type="audio/wav")


class TTSInputs(BaseModel):
    text: str
    language: str = None
    speaker_embedding: List[float] = None
    gpt_cond_latent: List[List[float]] = None
    speaker_idx: str = None  # For VITS models
    speed: float = 1.0  # Speed control: 0.5 = slower, 2.0 = faster


@app.post("/tts")
def predict_speech(parsed_input: TTSInputs):
    """Generate speech from text"""
    try:
        # Prepare TTS arguments
        tts_kwargs = {"text": parsed_input.text}

        # Add speaker if specified
        if parsed_input.speaker_idx is not None:
            tts_kwargs["speaker"] = parsed_input.speaker_idx

        # Add language if specified (mainly for XTTS)
        if parsed_input.language is not None:
            tts_kwargs["language"] = parsed_input.language

        # Add speed if specified
        if parsed_input.speed is not None:
            tts_kwargs["speed"] = parsed_input.speed

        # Generate speech
        out = model.tts(**tts_kwargs)

        # Convert to audio format
        wav = np.array(out)
        wav = np.clip(wav, -1, 1)
        wav = (wav * 32767).astype(np.int16)

        return encode_audio_common(wav.tobytes())

    except Exception as e:
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")


@app.get("/studio_speakers")
def get_speakers():
    """Get available speakers for the current model"""
    if hasattr(model, "speakers") and model.speakers:
        return {name: {"speaker_idx": name} for name in model.speakers}
    return {}


@app.get("/languages")
def get_languages():
    """Get supported languages for the current model"""
    return model.languages if hasattr(model, "languages") else []


@app.get("/model_info")
def get_model_info():
    """Get comprehensive information about the loaded model"""
    info = {
        "model_type": MODEL_TYPE.upper(),
        "model_name": default_model_name,
        "device": str(device),
        "supports_streaming": True,
        "supports_voice_cloning": MODEL_TYPE == "xtts",
        "languages": model.languages if hasattr(model, "languages") else [],
        "speakers": (
            list(model.speakers.keys())
            if hasattr(model, "speakers") and model.speakers
            else []
        ),
    }

    # Add model-specific information
    if MODEL_TYPE == "xtts":
        info.update(
            {
                "note": "Voice cloning requires raw XTTS model loading, not available via TTS API"
            }
        )
    elif MODEL_TYPE == "vits":
        info.update(
            {
                "num_speakers": (
                    len(model.speakers)
                    if hasattr(model, "speakers") and model.speakers
                    else 0
                ),
                "is_multi_speaker": (
                    len(model.speakers) > 1
                    if hasattr(model, "speakers") and model.speakers
                    else False
                ),
            }
        )

    return info
