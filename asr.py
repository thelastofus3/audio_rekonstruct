"""
# file: asr.py
Automatic Speech Recognition using Whisper (faster-whisper preferred, openai-whisper fallback).
Returns segments with timestamps, text, and confidence scores.
"""

import logging
import os
import time
from pathlib import Path
from typing import List, Optional


def run_asr(
    audio_path: str,
    model_name: str = "medium",
    language: Optional[str] = None,
    device: str = "auto",
    vad_segments: Optional[List[dict]] = None,
    confidence_threshold: float = 0.5,
    logger: logging.Logger = None,
) -> List[dict]:
    """
    Run ASR on audio file.

    Args:
        audio_path: Path to WAV file.
        model_name: Whisper model size.
        language: Language code or None for auto-detection.
        device: 'auto', 'cpu', or 'cuda'.
        vad_segments: Optional VAD segments to process selectively.
        confidence_threshold: Below this → mark as unclear.
        logger: Optional logger.

    Returns:
        List of segment dicts with keys:
          id, start, end, text, confidence, words, is_unclear
    """
    log = logger or logging.getLogger(__name__)

    # Resolve device
    if device == "auto":
        try:
            import torch
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            resolved_device = "cpu"
    else:
        resolved_device = device

    log.info(f"ASR device: {resolved_device}")

    # Try faster-whisper first
    try:
        return _asr_faster_whisper(
            audio_path,
            model_name=model_name,
            language=language,
            device=resolved_device,
            vad_segments=vad_segments,
            confidence_threshold=confidence_threshold,
            log=log,
        )
    except ImportError:
        log.info("faster-whisper not available, trying openai-whisper...")
    except Exception as e:
        log.warning(f"faster-whisper failed: {e}, trying openai-whisper...")

    # Fallback to openai-whisper
    try:
        return _asr_openai_whisper(
            audio_path,
            model_name=model_name,
            language=language,
            device=resolved_device,
            vad_segments=vad_segments,
            confidence_threshold=confidence_threshold,
            log=log,
        )
    except ImportError:
        log.error("Neither faster-whisper nor openai-whisper is installed!")
        raise ImportError(
            "Please install whisper: pip install faster-whisper\n"
            "or: pip install openai-whisper"
        )


# ─────────────────────────────────────────────────────────────────────────────
# faster-whisper
# ─────────────────────────────────────────────────────────────────────────────

def _asr_faster_whisper(
    audio_path: str,
    model_name: str,
    language: Optional[str],
    device: str,
    vad_segments: Optional[List[dict]],
    confidence_threshold: float,
    log: logging.Logger,
) -> List[dict]:
    from faster_whisper import WhisperModel

    log.info(f"Loading faster-whisper model '{model_name}' on {device}...")
    start = time.time()

    compute_type = "float16" if device == "cuda" else "int8"

    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
        download_root=str(Path.home() / ".cache" / "whisper"),
    )
    log.info(f"Model loaded in {time.time()-start:.1f}s")

    # Transcribe
    log.info("Transcribing...")
    start = time.time()

    use_builtin_vad = _should_use_builtin_vad(audio_path, vad_segments, log)

    transcribe_kwargs = {
        "word_timestamps": True,
        "vad_filter": use_builtin_vad,
        "vad_parameters": {"min_silence_duration_ms": 300},  # More sensitive to pauses
        "beam_size": 5,
        "best_of": 5,
        "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.5,  # Lower threshold = less likely to mark as no-speech
        "condition_on_previous_text": True,
        "initial_prompt": _get_initial_prompt(language),
        "repetition_penalty": 1.0,  # Prevent repetitions
        "no_repeat_ngram_size": 0,
    }

    if language:
        transcribe_kwargs["language"] = language

    segments_gen, info = model.transcribe(audio_path, **transcribe_kwargs)

    detected_lang = info.language
    lang_prob = info.language_probability
    log.info(
        f"Detected language: {detected_lang} (confidence={lang_prob:.2f})"
        + (f", forced={language}" if language else "")
    )

    if vad_segments:
        log.info(f"Using external VAD: {len(vad_segments)} segments provided")
    log.info(f"Whisper VAD filter: {'enabled' if use_builtin_vad else 'disabled'}")

    # Collect segments
    result_segments = []
    segment_id = 0

    for seg in segments_gen:
        # Compute confidence from avg_logprob
        # avg_logprob is typically in [-2.0, 0.0], map to [0, 1]
        log_prob = seg.avg_logprob if hasattr(seg, "avg_logprob") else -0.5
        confidence = _logprob_to_confidence(log_prob)

        # No-speech probability
        no_speech_prob = seg.no_speech_prob if hasattr(seg, "no_speech_prob") else 0.0

        # Determine if unclear
        is_unclear = (
            confidence < confidence_threshold
            or no_speech_prob > 0.7
            or len(seg.text.strip()) == 0
        )

        # Extract word-level timestamps
        words = []
        if hasattr(seg, "words") and seg.words:
            for w in seg.words:
                word_confidence = _logprob_to_confidence(
                    w.probability if hasattr(w, "probability") else None,
                    is_probability=True,
                )
                words.append({
                    "word": w.word,
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "confidence": round(word_confidence, 3),
                })

        text = seg.text.strip()

        # Apply VAD filter: skip segments outside VAD windows
        if vad_segments and not _segment_in_vad(seg.start, seg.end, vad_segments):
            log.debug(f"Skipping ASR segment outside VAD: {seg.start:.1f}-{seg.end:.1f}s")
            continue

        result_segments.append({
            "id": segment_id,
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": text,
            "confidence": round(confidence, 3),
            "no_speech_prob": round(no_speech_prob, 3),
            "words": words,
            "is_unclear": is_unclear,
            "language": detected_lang,
        })
        segment_id += 1

        # Progress log
        if segment_id % 20 == 0:
            log.info(f"  Transcribed {segment_id} segments... ({seg.end:.0f}s)")

    log.info(f"faster-whisper transcription done: {len(result_segments)} segments in {time.time()-start:.1f}s")
    return result_segments


# ─────────────────────────────────────────────────────────────────────────────
# openai-whisper
# ─────────────────────────────────────────────────────────────────────────────

def _asr_openai_whisper(
    audio_path: str,
    model_name: str,
    language: Optional[str],
    device: str,
    vad_segments: Optional[List[dict]],
    confidence_threshold: float,
    log: logging.Logger,
) -> List[dict]:
    import whisper
    import torch
    import numpy as np

    log.info(f"Loading openai-whisper model '{model_name}'...")
    start = time.time()

    model = whisper.load_model(
        model_name,
        device=device,
        download_root=str(Path.home() / ".cache" / "whisper"),
    )
    log.info(f"Model loaded in {time.time()-start:.1f}s")

    log.info("Transcribing...")
    start = time.time()

    transcribe_kwargs = {
        "word_timestamps": True,
        "verbose": None,
        "beam_size": 5,
        "best_of": 5,
        "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": True,
        "initial_prompt": _get_initial_prompt(language),
    }

    if language:
        transcribe_kwargs["language"] = language

    result = model.transcribe(audio_path, **transcribe_kwargs)

    detected_lang = result.get("language", language or "unknown")
    log.info(f"Detected language: {detected_lang}")

    result_segments = []
    for seg_id, seg in enumerate(result.get("segments", [])):
        log_prob = seg.get("avg_logprob", -0.5)
        confidence = _logprob_to_confidence(log_prob)
        no_speech_prob = seg.get("no_speech_prob", 0.0)

        is_unclear = (
            confidence < confidence_threshold
            or no_speech_prob > 0.7
            or len(seg.get("text", "").strip()) == 0
        )

        # Word timestamps
        words = []
        for w in seg.get("words", []):
            word_prob = w.get("probability", 0.5)
            words.append({
                "word": w.get("word", ""),
                "start": round(w.get("start", seg["start"]), 3),
                "end": round(w.get("end", seg["end"]), 3),
                "confidence": round(float(word_prob), 3),
            })

        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", 0.0)

        if vad_segments and not _segment_in_vad(seg_start, seg_end, vad_segments):
            continue

        result_segments.append({
            "id": seg_id,
            "start": round(seg_start, 3),
            "end": round(seg_end, 3),
            "text": seg.get("text", "").strip(),
            "confidence": round(confidence, 3),
            "no_speech_prob": round(no_speech_prob, 3),
            "words": words,
            "is_unclear": is_unclear,
            "language": detected_lang,
        })

    log.info(f"openai-whisper done: {len(result_segments)} segments in {time.time()-start:.1f}s")
    return result_segments


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _logprob_to_confidence(
    log_prob,
    is_probability: bool = False,
) -> float:
    """Convert log probability to [0, 1] confidence."""
    import math

    if is_probability:
        if log_prob is None:
            return 0.5
        return max(0.0, min(1.0, float(log_prob)))

    if log_prob is None:
        return 0.5

    lp = float(log_prob)
    # avg_logprob typically in [-2.0, 0.0]
    # Map: 0.0 -> 1.0,  -1.0 -> 0.37,  -2.0 -> 0.14
    confidence = math.exp(max(lp, -5.0))
    return round(max(0.0, min(1.0, confidence)), 4)


def _segment_in_vad(
    seg_start: float,
    seg_end: float,
    vad_segments: List[dict],
    overlap_threshold: float = 0.3,
) -> bool:
    """Check if ASR segment overlaps sufficiently with any VAD segment."""
    seg_duration = seg_end - seg_start
    if seg_duration <= 0:
        return False

    for vad in vad_segments:
        overlap_start = max(seg_start, vad["start"])
        overlap_end = min(seg_end, vad["end"])
        overlap = max(0.0, overlap_end - overlap_start)

        if overlap / seg_duration >= overlap_threshold:
            return True

    return False


def _should_use_builtin_vad(
    audio_path: str,
    vad_segments: Optional[List[dict]],
    log: logging.Logger,
) -> bool:
    """Prefer Whisper VAD when external VAD is missing or too coarse."""
    if not vad_segments:
        return True

    try:
        import scipy.io.wavfile as wav_io

        sr, data = wav_io.read(audio_path)
        audio_duration = len(data) / max(sr, 1)
    except Exception:
        audio_duration = None

    total_vad = sum(
        max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))
        for seg in vad_segments
    )

    if len(vad_segments) <= 1:
        log.info("External VAD is too coarse; keeping faster-whisper built-in VAD enabled.")
        return True

    if audio_duration and total_vad >= audio_duration * 0.9:
        log.info("External VAD covers nearly full audio; keeping faster-whisper built-in VAD enabled.")
        return True

    return False


def _get_initial_prompt(language: Optional[str]) -> Optional[str]:
    """Get language-specific initial prompt for better accuracy."""
    # Removed prompts to avoid Whisper hallucinations
    # Initial prompts can cause Whisper to return the prompt text instead of actual transcription
    return None
