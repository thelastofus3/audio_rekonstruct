# Speech Restoration Pipeline

Полный пайплайн восстановления речи из видео низкого качества.

## Что делает

1. **Извлекает аудио** из видео (mp4/mkv/avi/mov) через ffmpeg
2. **Улучшает аудио** — шумоподавление, дереверберация (denoiser / SpeechBrain / noisereduce)
3. **VAD** — определяет сегменты речи (Silero VAD / webrtcvad / energy-based)
4. **ASR** — транскрибирует с временными метками (faster-whisper / openai-whisper)
5. **Постобработка** — помечает неразборчивые места, опционально восстанавливает через LLM
6. **Диаризация** (опц.) — определяет говорящих (pyannote.audio)

## Выходные файлы

```
out_folder/
├── audio_raw.wav          # извлечённое аудио
├── audio_enhanced.wav     # улучшенное аудио
├── transcript.txt         # текст с временными метками
├── transcript.json        # JSON с confidence и метаданными
└── process_YYYYMMDD.log   # лог процесса
```

## Теги в транскрипте

| Тег | Значение |
|-----|----------|
| `[неразборчиво]` | Сегмент не распознан (слишком низкое качество) |
| `[предположение] текст` | Текст восстановлен, но уверенности мало |
| `[вариант1/вариант2]` | Возможны два варианта (от LLM) |
| `[слово?]` | Отдельное слово с низким confidence |

## Установка

### 1. Системные зависимости

**Windows:**
```
winget install Gyan.FFmpeg
```
или скачать ffmpeg.exe с https://ffmpeg.org/download.html и добавить в PATH.

**Linux/macOS:**
```bash
sudo apt install ffmpeg   # Ubuntu/Debian
brew install ffmpeg        # macOS
```

### 2. Python зависимости

```bash
pip install -r requirements.txt
```

Для GPU ускорения (CUDA):
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Опциональные зависимости

**SpeechBrain (лучше качество шумоподавления):**
```bash
pip install speechbrain
```

**facebook/denoiser:**
```bash
pip install denoiser
```

**Диаризация говорящих:**
```bash
pip install pyannote.audio
# Получить токен на https://huggingface.co/pyannote/speaker-diarization-3.1
# Принять условия использования модели
export HF_TOKEN=your_token_here
```

**LLM постобработка:**
```bash
pip install openai   # для GPT-4o
# export OPENAI_API_KEY=sk-...

pip install anthropic  # для Claude
# export ANTHROPIC_API_KEY=sk-ant-...
```

## Использование

### Базовый запуск

```bash
python main.py --input video.mp4 --output ./results --language ru
```

### Полный набор опций

```bash
python main.py \
  --input video.mp4 \
  --output ./results \
  --language ru \
  --model medium \
  --device auto
```

### С диаризацией

```bash
python main.py --input interview.mp4 --output ./out --language ru --diarize
```

### С LLM постобработкой

```bash
export OPENAI_API_KEY=sk-...
python main.py --input lecture.mp4 --output ./out --language ru --llm-postprocess --llm-provider openai
```

### Без улучшения аудио (быстрый режим)

```bash
python main.py --input video.mp4 --output ./out --language ru --no-enhance
```

## Параметры CLI

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `--input` | — | Входной видеофайл (обязательно) |
| `--output` | — | Папка для результатов (обязательно) |
| `--language` | `auto` | Язык: ru, en, de, fr, uk, auto |
| `--model` | `medium` | Whisper: tiny/base/small/medium/large/large-v2/large-v3 |
| `--device` | `auto` | Устройство: auto/cpu/cuda |
| `--diarize` | off | Включить диаризацию говорящих |
| `--no-enhance` | off | Пропустить улучшение аудио |
| `--no-vad` | off | Пропустить VAD |
| `--llm-postprocess` | off | Восстановление через LLM |
| `--llm-provider` | `openai` | openai / anthropic / local |
| `--confidence-threshold` | `0.5` | Порог confidence (0.0-1.0) |

## Выбор модели Whisper

| Модель | VRAM | Скорость | Качество |
|--------|------|----------|----------|
| tiny | ~1 GB | очень быстро | низкое |
| base | ~1 GB | быстро | среднее |
| small | ~2 GB | хорошо | хорошее |
| **medium** | ~5 GB | умеренно | **рекомендуется** |
| large-v2 | ~10 GB | медленно | отличное |
| large-v3 | ~10 GB | медленно | лучшее |

## Формат transcript.json

```json
{
  "created_at": "2024-01-15T10:30:00",
  "total_segments": 42,
  "segments": [
    {
      "id": 0,
      "start": 1.234,
      "end": 4.567,
      "text": "Добрый день уважаемые коллеги",
      "confidence": 0.87,
      "no_speech_prob": 0.02,
      "is_unclear": false,
      "restoration_tag": null,
      "language": "ru",
      "speaker": "SPEAKER_00",
      "words": [
        {"word": "Добрый", "start": 1.234, "end": 1.567, "confidence": 0.95},
        {"word": "день", "start": 1.600, "end": 1.890, "confidence": 0.91}
      ]
    },
    {
      "id": 5,
      "start": 12.100,
      "end": 14.500,
      "text": "[предположение] важный вопрос повестки",
      "confidence": 0.38,
      "is_unclear": true,
      "restoration_tag": "low_confidence"
    },
    {
      "id": 8,
      "start": 22.000,
      "end": 23.500,
      "text": "[неразборчиво]",
      "confidence": 0.12,
      "is_unclear": true,
      "restoration_tag": "inaudible"
    }
  ],
  "full_text": "..."
}
```

## Структура проекта

```
speech_restore/
├── main.py           # CLI точка входа, оркестрация пайплайна
├── extract_audio.py  # Извлечение аудио через ffmpeg
├── enhance_audio.py  # Улучшение аудио (denoiser/SpeechBrain/noisereduce)
├── vad.py            # Voice Activity Detection
├── asr.py            # Распознавание речи (Whisper)
├── postprocess.py    # Постобработка транскрипта, LLM восстановление
├── diarize.py        # Диаризация говорящих (опционально)
├── requirements.txt
└── README.md
```

## Troubleshooting

**`ffmpeg not found`** — установите ffmpeg и добавьте в PATH.

**`CUDA out of memory`** — используйте меньшую модель (`--model small`) или `--device cpu`.

**`webrtcvad` не устанавливается на Windows** — используйте `webrtcvad-wheels` (уже в requirements.txt).

**Плохое качество транскрипции** — попробуйте `--model large-v3` или `--no-enhance` если улучшение ухудшает аудио.

**Медленная работа на CPU** — используйте `--model base` или `--model small` для быстрого результата.
