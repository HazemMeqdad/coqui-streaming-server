# Universal TTS Server
*Text-to-Speech server supporting both XTTS and VITS models with automatic detection using TTS library*

This server provides a unified REST API for text-to-speech synthesis that automatically detects the model type and adapts its functionality accordingly. Built with the high-level TTS library for simplified model management.

## Supported Models

### XTTS Models
- **Full name**: `tts_models/multilingual/multi-dataset/xtts_v2` (default)
- **Features**: Multilingual support, streaming inference
- **Languages**: Multiple (en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi)
- **Speakers**: Pre-defined studio speakers (if available)

### VITS Models  
- **Example**: `tts_models/en/vctk/vits`
- **Features**: High-quality single or multi-speaker synthesis
- **Languages**: Typically single language per model
- **Speakers**: Pre-defined speaker voices (for multi-speaker models)

## Features

- **Automatic Model Detection**: Server automatically detects XTTS vs VITS based on model name
- **Simplified TTS API**: Uses high-level TTS library for easy model management
- **Universal Endpoints**: Same endpoints work with both model types
- **Multi-speaker Support**: Both model types support multiple speakers via speaker selection
- **Streaming Support**: Audio streaming for better user experience
- **REST API**: Easy-to-use HTTP endpoints
- **Error Handling**: Robust error handling and validation

## Quick Start

### Using XTTS (Default)
```bash
# Use default XTTS model
USE_CPU=1 uvicorn server.main:app --host 0.0.0.0 --port 8888

# Or with Docker
docker run --rm -p 8888:8888 -e DEFAULT_MODEL_NAME="tts_models/multilingual/multi-dataset/xtts_v2" ghcr.io/hazemmeqdad/coqui-streaming-server:latest
```

### Using VITS
```bash
# Use VITS model
USE_CPU=1 DEFAULT_MODEL_NAME=tts_models/en/vctk/vits uvicorn server.main:app --host 0.0.0.0 --port 8888

# Or with Docker
docker run --rm -p 8888:8888 -e DEFAULT_MODEL_NAME="tts_models/en/vctk/vits" ghcr.io/hazemmeqdad/coqui-streaming-server:latest
```

## Environment Variables

- `DEFAULT_MODEL_NAME`: Model to load (default: `tts_models/multilingual/multi-dataset/xtts_v2`)
- `USE_CPU`: Set to "1" to force CPU usage (default: "0", uses GPU if available)
- `NUM_THREADS`: Number of CPU threads (default: auto-detect)

## API Endpoints

### Universal Endpoints (work with both model types)

#### `POST /tts`
Generate speech from text. Parameters adapt based on model type:

**For VITS models:**
```json
{
  "text": "Hello world",
  "speaker_idx": "p225"
}
```

**For XTTS models:**
```json
{
  "text": "Hello world",
  "language": "en"
}
```

#### `POST /tts_stream`
Stream audio generation (same parameters as `/tts` plus streaming options):

```json
{
  "text": "Hello world",
  "speaker_idx": "p225",
  "add_wav_header": true,
  "stream_chunk_size": "20"
}
```

#### `GET /studio_speakers`
Get available speakers:

```json
{
  "p225": {"speaker_idx": "p225"},
  "p226": {"speaker_idx": "p226"}
}
```

#### `GET /languages` 
Get supported languages:

```json
["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"]
```

#### `GET /model_info`
Get comprehensive model information:

```json
{
  "model_type": "XTTS",
  "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
  "device": "cpu",
  "supports_streaming": true,
  "supports_voice_cloning": false,
  "languages": ["en", "es", ...],
  "speakers": ["speaker_0", "speaker_1", ...]
}
```

## Usage Examples

### Python Client
```python
import requests
import base64

# Check what model is loaded
model_info = requests.get("http://localhost:8888/model_info").json()
print(f"Model type: {model_info['model_type']}")
print(f"Available speakers: {model_info['speakers']}")
print(f"Available languages: {model_info['languages']}")

# Prepare request based on model type
if model_info['model_type'] == 'XTTS':
    payload = {
        "text": "Hello, this is XTTS speech!",
        "language": "en"
    }
else:
    # VITS usage with speaker selection
    speakers = requests.get("http://localhost:8888/studio_speakers").json()
    first_speaker = list(speakers.keys())[0] if speakers else None
    
    payload = {
        "text": "Hello, this is VITS speech!",
        "speaker_idx": first_speaker
    }

# Generate speech
response = requests.post("http://localhost:8888/tts", json=payload)
audio_data = base64.b64decode(response.content)

with open("output.wav", "wb") as f:
    f.write(audio_data)
```

### cURL Examples
```bash
# Check model info
curl http://localhost:8888/model_info

# Get available speakers
curl http://localhost:8888/studio_speakers

# Generate speech with VITS
curl -X POST http://localhost:8888/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "speaker_idx": "p225"}' \
  --output speech.wav

# Generate speech with XTTS
curl -X POST http://localhost:8888/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "language": "en"}' \
  --output speech.wav

# Stream audio
curl -X POST http://localhost:8888/tts_stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "speaker_idx": "p225"}' \
  --output stream.wav
```

## Model Comparison

| Feature | XTTS | VITS |
|---------|------|------|
| Voice Cloning | ❌ (not available via TTS API) | ❌ |
| Streaming | ✅ | ✅ |
| Multilingual | ✅ | ❌ (single language) |
| Speaker Selection | ✅ | ✅ |
| Quality | High | Very High |
| Speed | Fast | Very Fast |
| GPU Memory | Higher | Lower |
| Use Case | Multilingual synthesis | High-quality single-language synthesis |

## Demo Interface

The server includes a Gradio web interface that automatically adapts to the loaded model:

```bash
python demo.py
```

Access at http://localhost:3009

## Docker Support

Available Docker images for different configurations:

```bash
# CPU-only with XTTS
docker run --rm -p 8888:8888 -e USE_CPU=1 ghcr.io/hazemmeqdad/coqui-streaming-server:latest

# GPU with XTTS (CUDA 12.1)  
docker run --rm --gpus all -p 8888:8888 ghcr.io/hazemmeqdad/coqui-streaming-server:latest

# VITS model
docker run --rm -p 8888:8888 -e DEFAULT_MODEL_NAME="tts_models/en/vctk/vits" ghcr.io/hazemmeqdad/coqui-streaming-server:latest
```

## Development

### Local Setup
```bash
# Install dependencies
pip install -r server/requirements.txt

# Run server with XTTS (default)
cd server && USE_CPU=1 uvicorn main:app --host 0.0.0.0 --port 8888 --reload

# Run server with VITS
cd server && USE_CPU=1 DEFAULT_MODEL_NAME=tts_models/en/vctk/vits uvicorn main:app --host 0.0.0.0 --port 8888 --reload

# Run demo interface
pip install gradio  # if not already installed
python demo.py
```

### Testing
```bash
# Test model info
curl http://localhost:8888/model_info

# Test speakers
curl http://localhost:8888/studio_speakers

# Test languages  
curl http://localhost:8888/languages

# Test TTS
curl -X POST http://localhost:8888/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world test"}' \
  --output test.wav
```

## Notes

- **Voice Cloning**: The original voice cloning functionality requires raw XTTS model loading and is not available through the simplified TTS API. This trade-off was made for easier model management and broader model support.

- **Streaming**: While both models support the `/tts_stream` endpoint, true real-time streaming is model-dependent.

- **Model Loading**: The server automatically downloads models on first use. Subsequent starts will be faster.

- **Memory Usage**: XTTS models require more memory than VITS models. Use CPU mode (`USE_CPU=1`) if you encounter memory issues.

## License

This project is licensed under the AGPL-3.0 License - see the [LICENSE](LICENSE) file for details.
