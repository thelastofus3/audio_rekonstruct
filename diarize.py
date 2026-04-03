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

    methods = [
        ("pyannote.audio", _diarize_pyannote),
        ("simple energy-based", _diarize_simple),
    ]

    diarization_result = None
    for method_name, method_fn in methods:
        try:
            log.info(f"Trying {method_name}...")
            diarization_result = method_fn(
                audio_path,
                hf_token=hf_token,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                log=log,
            )
            if diarization_result:
                log.info(f"Diarization successful: {method_name}")
                break
        except ImportError as e:
            log.info(f"{method_name} not available: {e}")
        except Exception as e:
            log.warning(f"{method_name} failed: {e}")

    if not diarization_result:
        log.warning("Diarization failed. Segments will have no speaker labels.")
        return segments

    # Assign speakers to segments
    return _assign_speakers(segments, diarization_result, log)


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

    # Convert to list of dicts
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


def _diarize_simple(
    audio_path: str,
    hf_token: str,
    min_speakers: int,
    max_speakers: int,
    log: logging.Logger,
) -> List[dict]:
    """
    Simple heuristic diarization based on spectral features.
    Not accurate but works without external dependencies.
    """
    import numpy as np
    import scipy.io.wavfile as wav_io
    import scipy.signal
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    log.info("Running simple spectral diarization...")

    sr, data = wav_io.read(audio_path)

    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    else:
        audio = data.astype(np.float32)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Extract spectral features per frame
    frame_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)

    features = []
    timestamps = []

    for i in range(0, len(audio) - frame_len, hop_len):
        frame = audio[i : i + frame_len]
        energy = np.sum(frame ** 2)

        if energy < 1e-6:  # silence
            continue

        # Compute spectral centroid and rolloff
        spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
        freqs = np.fft.rfftfreq(len(frame), 1/sr)

        total_power = spectrum.sum() + 1e-10
        centroid = (freqs * spectrum).sum() / total_power
        rolloff_idx = np.searchsorted(np.cumsum(spectrum), 0.85 * spectrum.sum())
        rolloff = freqs[min(rolloff_idx, len(freqs)-1)]

        features.append([centroid, rolloff, np.log(energy + 1e-10)])
        timestamps.append(i / sr)

    if len(features) < 10:
        log.warning("Not enough features for clustering.")
        return []

    features = np.array(features)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Determine number of clusters
    n_clusters = min(max(min_speakers, 2), max_speakers, len(features) // 50)
    n_clusters = max(n_clusters, 2)

    log.info(f"Clustering into {n_clusters} speakers...")
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(features_scaled)

    # Create speaker segments from labels
    speaker_segments = []
    prev_label = labels[0]
    seg_start = timestamps[0]

    for j in range(1, len(labels)):
        if labels[j] != prev_label:
            speaker_segments.append({
                "start": seg_start,
                "end": timestamps[j],
                "speaker": f"SPEAKER_{prev_label:02d}",
            })
            prev_label = labels[j]
            seg_start = timestamps[j]

    # Last segment
    speaker_segments.append({
        "start": seg_start,
        "end": timestamps[-1],
        "speaker": f"SPEAKER_{prev_label:02d}",
    })

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

        # Find diarization segment with most overlap
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

    # Log speaker distribution
    from collections import Counter
    speaker_counts = Counter(s["speaker"] for s in result)
    log.info("Speaker distribution: " + ", ".join(
        f"{sp}: {cnt}" for sp, cnt in speaker_counts.most_common()
    ))

    return result
