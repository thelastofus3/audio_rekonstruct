# Speech Restoration Pipeline

Full pipeline for restoring speech from low-quality video.

## What It Does

1. **Extracts audio** from video files (mp4/mkv/avi/mov) via ffmpeg
2. **Enhances audio** with noise reduction and dereverberation (denoiser / SpeechBrain / noisereduce)
3. **VAD** detects speech segments (Silero VAD / webrtcvad / energy-based)
4. **ASR** transcribes speech with timestamps (faster-whisper / openai-whisper)
5. **Postprocessing** marks unclear parts and can optionally restore them with an LLM
6. **Diarization** (optional) identifies speakers (pyannote.audio)

## Output Files

```
out_folder/
├── audio_raw.wav          # extracted audio
├── audio_enhanced.wav     # enhanced audio
├── transcript.txt         # transcript with timestamps
├── transcript.json        # JSON with confidence and metadata
└── process_YYYYMMDD.log   # process log
```

## Transcript Tags

| Tag | Meaning |
|-----|---------|
| `[inaudible]` | Segment could not be recognized due to very low quality |
| `[assumption] text` | Restored text with low confidence |
| `[variant1/variant2]` | Two possible variants suggested by the LLM |
| `[word?]` | Individual word with low confidence |

## Installation

### 1. System Dependencies

**Windows:**
```
winget install Gyan.FFmpeg
```
Or download `ffmpeg.exe` from https://ffmpeg.org/download.html and add it to `PATH`.

**Linux/macOS:**
```bash
sudo apt install ffmpeg   # Ubuntu/Debian
brew install ffmpeg       # macOS
```

### 2. Python Dependencies

```bash
pip install -r requirements.txt
```

For GPU acceleration (CUDA):
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Optional Dependencies

**SpeechBrain (better denoising quality):**
```bash
pip install speechbrain
```

**facebook/denoiser:**
```bash
pip install denoiser
```

**Speaker diarization:**
```bash
pip install pyannote.audio
# Get a token from https://huggingface.co/pyannote/speaker-diarization-3.1
# Accept the model terms of use
export HF_TOKEN=your_token_here
```

**LLM postprocessing:**
```bash
pip install openai   # for GPT-4o
# export OPENAI_API_KEY=sk-...

pip install anthropic  # for Claude
# export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

### Basic Run

```bash
python main.py --input video.mp4 --output ./results --language ru
```

### Full Set of Options

```bash
python main.py \
  --input video.mp4 \
  --output ./results \
  --language ru \
  --model medium \
  --device auto
```

### With Diarization

```bash
python main.py --input interview.mp4 --output ./out --language ru --diarize
```

### With LLM Postprocessing

```bash
export OPENAI_API_KEY=sk-...
python main.py --input lecture.mp4 --output ./out --language ru --llm-postprocess --llm-provider openai
```

### Without Audio Enhancement (Fast Mode)

```bash
python main.py --input video.mp4 --output ./out --language ru --no-enhance
```

## CLI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | — | Input video file (required) |
| `--output` | — | Output directory for results (required) |
| `--language` | `auto` | Language: ru, en, de, fr, uk, auto |
| `--model` | `medium` | Whisper model: tiny/base/small/medium/large/large-v2/large-v3 |
| `--device` | `auto` | Device: auto/cpu/cuda |
| `--diarize` | off | Enable speaker diarization |
| `--no-enhance` | off | Skip audio enhancement |
| `--no-vad` | off | Skip VAD |
| `--llm-postprocess` | off | Restore unclear text with an LLM |
| `--llm-provider` | `openai` | `openai` / `anthropic` / `local` |
| `--confidence-threshold` | `0.5` | Confidence threshold (0.0-1.0) |

## Whisper Model Selection

| Model | VRAM | Speed | Quality |
|-------|------|-------|---------|
| tiny | ~1 GB | very fast | low |
| base | ~1 GB | fast | medium |
| small | ~2 GB | good | good |
| **medium** | ~5 GB | moderate | **recommended** |
| large-v2 | ~10 GB | slow | excellent |
| large-v3 | ~10 GB | slow | best |

## `transcript.json` Format

```json
{
  "created_at": "2024-01-15T10:30:00",
  "total_segments": 42,
  "segments": [
    {
      "id": 0,
      "start": 1.234,
      "end": 4.567,
      "text": "Good afternoon, dear colleagues",
      "confidence": 0.87,
      "no_speech_prob": 0.02,
      "is_unclear": false,
      "restoration_tag": null,
      "language": "ru",
      "speaker": "SPEAKER_00",
      "words": [
        {"word": "Good", "start": 1.234, "end": 1.567, "confidence": 0.95},
        {"word": "afternoon", "start": 1.600, "end": 1.890, "confidence": 0.91}
      ]
    },
    {
      "id": 5,
      "start": 12.100,
      "end": 14.500,
      "text": "[assumption] important agenda item",
      "confidence": 0.38,
      "is_unclear": true,
      "restoration_tag": "low_confidence"
    },
    {
      "id": 8,
      "start": 22.000,
      "end": 23.500,
      "text": "[inaudible]",
      "confidence": 0.12,
      "is_unclear": true,
      "restoration_tag": "inaudible"
    }
  ],
  "full_text": "..."
}
```

## Project Structure

```
speech_restore/
├── main.py           # CLI entry point, pipeline orchestration
├── extract_audio.py  # Audio extraction via ffmpeg
├── enhance_audio.py  # Audio enhancement (denoiser/SpeechBrain/noisereduce)
├── vad.py            # Voice Activity Detection
├── asr.py            # Speech recognition (Whisper)
├── postprocess.py    # Transcript postprocessing and LLM restoration
├── diarize.py        # Speaker diarization (optional)
├── requirements.txt
└── README.md
```

## Troubleshooting

**`ffmpeg not found`**: install `ffmpeg` and add it to `PATH`.

**`CUDA out of memory`**: use a smaller model such as `--model small` or switch to `--device cpu`.

**`webrtcvad` fails to install on Windows**: use `webrtcvad-wheels` (already listed in `requirements.txt`).

**Poor transcription quality**: try `--model large-v3`, or `--no-enhance` if enhancement is degrading the audio.

**Slow performance on CPU**: use `--model base` or `--model small` for faster results.
