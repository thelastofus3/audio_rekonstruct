"""
# file: diarize.py
Optional speaker diarization using pyannote.audio.
Assigns speaker labels to ASR segments.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional


def run_diarization(
    audio_path: str,
    segments: List[dict],
    hf_token: Optional[str] = None,
    logger: logging.Logger = None,
    min_speakers: int = 1,
    max_speakers: int = 10,
) -> List[dict]:
    """
    Run speaker diarization and add speaker labels to segments.

    Args:
        audio_path: Path to WAV file.
        segments: ASR segments list.
        hf_token: HuggingFace token (required for pyannote.audio).
        logger: Optional logger.
        min_speakers: Minimum expected speakers.
        max_speakers: Maximum expected speakers.

    Returns:
        Segments with 'speaker' field added.
    """
    log = logger or logging.getLogger(__name__)

    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

    if not hf_token:
        log.warning(
            "No HuggingFace token found. Diarization requires pyannote.audio access.\n"
            "Set HF_TOKEN environment variable or pass --hf-token.\n"
            "Skipping diarization."
        )
        return segments

    log.info("Attempting speaker diarization...")

    try:
        diarization_result = _diarize_pyannote(
            audio_path,
            hf_token=hf_token,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            log=log,
        )
        if diarization_result:
            return _assign_speakers(segments, diarization_result, log)
    except ImportError as e:
        log.info(f"pyannote.audio not available: {e}")
    except Exception as e:
        log.warning(f"pyannote.audio diarization failed: {e}")

    log.warning("Diarization failed. Segments will have no speaker labels.")
    return segments


def _diarize_pyannote(
    audio_path: str,
    hf_token: str,
    min_speakers: int,
    max_speakers: int,
    log: logging.Logger,
) -> List[dict]:
    """Use pyannote.audio for diarization."""
    from pyannote.audio import Pipeline
    import torch

    log.info("Loading pyannote.audio speaker-diarization-3.1 pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = pipeline.to(device)

    log.info("Running diarization pipeline...")
    diarization = pipeline(
        audio_path,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
        })

    speakers = set(s["speaker"] for s in speaker_segments)
    log.info(f"Found {len(speakers)} speaker(s): {', '.join(sorted(speakers))}")

    return speaker_segments


def _assign_speakers(
    asr_segments: List[dict],
    diarization: List[dict],
    log: logging.Logger,
) -> List[dict]:
    """Assign speaker labels to ASR segments based on overlap."""
    result = []

    for seg in asr_segments:
        seg = dict(seg)
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", 0.0)
        seg_duration = max(seg_end - seg_start, 0.001)

        best_speaker = "UNKNOWN"
        best_overlap = 0.0

        for d_seg in diarization:
            overlap_start = max(seg_start, d_seg["start"])
            overlap_end = min(seg_end, d_seg["end"])
            overlap = max(0.0, overlap_end - overlap_start)
            overlap_ratio = overlap / seg_duration

            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_speaker = d_seg["speaker"]

        seg["speaker"] = best_speaker if best_overlap > 0.3 else "UNKNOWN"
        seg["speaker_confidence"] = round(best_overlap, 3)
        result.append(seg)

    from collections import Counter
    speaker_counts = Counter(s["speaker"] for s in result)
    log.info("Speaker distribution: " + ", ".join(
        f"{sp}: {cnt}" for sp, cnt in speaker_counts.most_common()
    ))

    return result
