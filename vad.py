"""
# file: vad.py
Voice Activity Detection — finds speech segments in audio.

Strategy (fallback chain):
  1. Silero VAD (best accuracy)
  2. webrtcvad (lightweight, fast)
  3. Energy-based VAD (no dependencies)
"""

import logging
import os
from pathlib import Path
from typing import List, Tuple


def run_vad(
    audio_path: str,
    logger: logging.Logger = None,
    aggressiveness: int = 2,
    min_speech_duration: float = 0.3,
    max_silence_duration: float = 0.8,
) -> List[dict]:
    """
    Run Voice Activity Detection on audio file.

    Args:
        audio_path: Path to WAV file (16kHz mono).
        logger: Optional logger.
        aggressiveness: 0-3 for webrtcvad (3 = most aggressive).
        min_speech_duration: Minimum speech segment duration in seconds.
        max_silence_duration: Max silence gap before splitting segment.

    Returns:
        List of dicts: [{"start": float, "end": float}, ...]
    """
    log = logger or logging.getLogger(__name__)

    methods = [
        ("Silero VAD", _vad_silero),
        ("webrtcvad", _vad_webrtc),
        ("Energy-based VAD", _vad_energy),
    ]

    for method_name, method_fn in methods:
        log.info(f"Trying {method_name}...")
        try:
            segments = method_fn(
                audio_path,
                min_speech_duration=min_speech_duration,
                max_silence_duration=max_silence_duration,
                log=log,
            )
            if segments:
                log.info(
                    f"VAD ({method_name}): {len(segments)} segments, "
                    f"total speech={_total_duration(segments):.1f}s"
                )
                return segments
        except ImportError as e:
            log.info(f"{method_name} not available: {e}")
        except Exception as e:
            log.warning(f"{method_name} failed: {e}")

    log.warning("All VAD methods failed — returning full audio as one segment.")
    return _get_full_duration_segment(audio_path)


# ─────────────────────────────────────────────────────────────────────────────
# Method 1: Silero VAD
# ─────────────────────────────────────────────────────────────────────────────

def _vad_silero(
    audio_path: str,
    min_speech_duration: float,
    max_silence_duration: float,
    log: logging.Logger,
) -> List[dict]:
    """Use Silero VAD for accurate voice detection."""
    import torch
    import torchaudio

    log.info("Loading Silero VAD model...")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        trust_repo=True,
    )

    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

    # Load and prepare audio
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.squeeze(0)

    # Resample to 16kHz if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        wav = resampler(wav)
        sr = 16000

    log.info("Running Silero VAD inference...")
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=sr,
        threshold=0.5,
        min_speech_duration_ms=int(min_speech_duration * 1000),
        min_silence_duration_ms=int(max_silence_duration * 1000),
        speech_pad_ms=200,
        return_seconds=True,
    )

    segments = [
        {"start": float(ts["start"]), "end": float(ts["end"])}
        for ts in speech_timestamps
    ]

    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Method 2: webrtcvad
# ─────────────────────────────────────────────────────────────────────────────

def _vad_webrtc(
    audio_path: str,
    min_speech_duration: float,
    max_silence_duration: float,
    log: logging.Logger,
    aggressiveness: int = 2,
) -> List[dict]:
    """Use webrtcvad for voice activity detection."""
    import webrtcvad
    import numpy as np
    import scipy.io.wavfile as wav_io

    vad = webrtcvad.Vad(aggressiveness)

    sr, data = wav_io.read(audio_path)

    # webrtcvad requires 8kHz, 16kHz, or 32kHz
    supported_rates = [8000, 16000, 32000]
    if sr not in supported_rates:
        # Resample to 16kHz
        import scipy.signal
        target_sr = 16000
        num_samples = int(len(data) * target_sr / sr)
        data = scipy.signal.resample(data, num_samples).astype(np.int16)
        sr = target_sr

    if data.dtype != np.int16:
        data = (data * 32767).astype(np.int16) if data.dtype == np.float32 else data.astype(np.int16)

    if data.ndim > 1:
        data = data[:, 0]

    # webrtcvad frame duration: 10, 20, or 30ms
    frame_duration_ms = 20
    frame_samples = int(sr * frame_duration_ms / 1000)

    log.info(f"Running webrtcvad (aggressiveness={aggressiveness})...")

    # Collect frame-level decisions
    frames = []
    for i in range(0, len(data) - frame_samples, frame_samples):
        frame = data[i : i + frame_samples]
        if len(frame) < frame_samples:
            break
        is_speech = vad.is_speech(frame.tobytes(), sample_rate=sr)
        timestamp = i / sr
        frames.append((timestamp, is_speech))

    # Merge frames into segments
    segments = _merge_vad_frames(
        frames,
        frame_duration_ms / 1000.0,
        min_speech_duration,
        max_silence_duration,
    )

    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Method 3: Energy-based VAD
# ─────────────────────────────────────────────────────────────────────────────

def _vad_energy(
    audio_path: str,
    min_speech_duration: float,
    max_silence_duration: float,
    log: logging.Logger,
) -> List[dict]:
    """Simple energy-based VAD — no external dependencies needed."""
    import numpy as np
    import scipy.io.wavfile as wav_io
    import scipy.signal

    log.info("Running energy-based VAD...")

    sr, data = wav_io.read(audio_path)

    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    else:
        audio = data.astype(np.float32)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Compute short-time energy
    frame_len = int(0.025 * sr)   # 25ms
    hop_len = int(0.010 * sr)     # 10ms

    energies = []
    timestamps = []

    for i in range(0, len(audio) - frame_len, hop_len):
        frame = audio[i : i + frame_len]
        energy = np.sum(frame ** 2) / frame_len
        energies.append(energy)
        timestamps.append(i / sr)

    energies = np.array(energies)

    # Adaptive threshold: use percentile-based thresholding
    # Assume top 10% of energy frames are speech
    noise_floor = np.percentile(energies, 30)
    speech_floor = np.percentile(energies, 70)
    threshold = noise_floor + 0.3 * (speech_floor - noise_floor)
    threshold = max(threshold, 1e-6)

    # Smooth decisions with median filter
    is_speech = energies > threshold
    from scipy.ndimage import median_filter
    is_speech = median_filter(is_speech.astype(float), size=5) > 0.5

    # Create frames list
    hop_duration = hop_len / sr
    frames = [(timestamps[i], bool(is_speech[i])) for i in range(len(timestamps))]

    segments = _merge_vad_frames(
        frames,
        hop_duration,
        min_speech_duration,
        max_silence_duration,
    )

    return segments


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _merge_vad_frames(
    frames: List[Tuple[float, bool]],
    frame_duration: float,
    min_speech_duration: float,
    max_silence_duration: float,
) -> List[dict]:
    """Merge frame-level VAD decisions into speech segments."""
    if not frames:
        return []

    segments = []
    in_speech = False
    speech_start = 0.0
    silence_start = 0.0

    for timestamp, is_speech in frames:
        if is_speech and not in_speech:
            speech_start = timestamp
            in_speech = True
            silence_start = 0.0
        elif not is_speech and in_speech:
            if silence_start == 0.0:
                silence_start = timestamp
            elif timestamp - silence_start >= max_silence_duration:
                # End of speech segment
                duration = silence_start - speech_start
                if duration >= min_speech_duration:
                    segments.append({
                        "start": speech_start,
                        "end": silence_start + frame_duration,
                    })
                in_speech = False
                silence_start = 0.0
        elif is_speech and in_speech:
            silence_start = 0.0  # reset silence counter

    # Handle last segment
    if in_speech:
        end = frames[-1][0] + frame_duration
        duration = end - speech_start
        if duration >= min_speech_duration:
            segments.append({"start": speech_start, "end": end})

    return segments


def _get_full_duration_segment(audio_path: str) -> List[dict]:
    """Return single segment covering the entire audio file."""
    try:
        import scipy.io.wavfile as wav_io
        sr, data = wav_io.read(audio_path)
        duration = len(data) / sr
        return [{"start": 0.0, "end": duration}]
    except Exception:
        return [{"start": 0.0, "end": 9999.0}]


def _total_duration(segments: List[dict]) -> float:
    return sum(s["end"] - s["start"] for s in segments)
