"""
# file: extract_audio.py
Extract audio from video using ffmpeg.
Output: 16kHz mono WAV (optimal for speech models).
"""

import logging
import os
import subprocess
import sys
from pathlib import Path


def extract_audio(
    video_path: str,
    output_wav: str,
    sample_rate: int = 16000,
    logger: logging.Logger = None,
) -> str:
    """
    Extract audio from video file using ffmpeg.

    Args:
        video_path: Path to input video file.
        output_wav: Path to output WAV file.
        sample_rate: Target sample rate (default 16000 Hz for speech models).
        logger: Optional logger instance.

    Returns:
        Path to the extracted WAV file.

    Raises:
        FileNotFoundError: If video file or ffmpeg not found.
        RuntimeError: If ffmpeg extraction fails.
    """
    log = logger or logging.getLogger(__name__)

    video_path = Path(video_path)
    output_wav = Path(output_wav)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Ensure output directory exists
    output_wav.parent.mkdir(parents=True, exist_ok=True)

    # Get video info first
    log.info(f"Probing video: {video_path.name}")
    probe_info = _probe_video(str(video_path), log)
    if probe_info:
        log.info(f"Video info: {probe_info}")

    # Build ffmpeg command
    # -ac 1       : mono
    # -ar <rate>  : resample to target sample rate
    # -acodec pcm_s16le : 16-bit PCM WAV
    # -vn         : no video
    # -af "aresample=resampler=soxr" : high-quality resampler if available
    cmd = [
        "ffmpeg",
        "-y",                          # overwrite output
        "-i", str(video_path),         # input
        "-vn",                         # no video
        "-ac", "1",                    # mono
        "-ar", str(sample_rate),       # sample rate
        "-acodec", "pcm_s16le",        # 16-bit PCM
        "-af", "aresample=resampler=soxr",  # high-quality resampler
        str(output_wav),
    ]

    log.info(f"Extracting audio: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    # If soxr not available, retry without it
    if result.returncode != 0 and "soxr" in result.stderr:
        log.warning("soxr resampler not available, retrying without it...")
        cmd_simple = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn", "-ac", "1",
            "-ar", str(sample_rate),
            "-acodec", "pcm_s16le",
            str(output_wav),
        ]
        result = subprocess.run(
            cmd_simple,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

    if result.returncode != 0:
        log.error(f"ffmpeg stderr:\n{result.stderr[-2000:]}")
        raise RuntimeError(
            f"ffmpeg failed with code {result.returncode}.\n"
            f"stderr: {result.stderr[-500:]}"
        )

    if not output_wav.exists() or output_wav.stat().st_size == 0:
        raise RuntimeError(f"Output WAV is empty or missing: {output_wav}")

    size_mb = output_wav.stat().st_size / (1024 * 1024)
    log.info(f"Audio extracted successfully: {output_wav} ({size_mb:.2f} MB)")

    # Log duration
    duration = _get_audio_duration(str(output_wav), log)
    if duration:
        log.info(f"Audio duration: {_format_duration(duration)}")

    return str(output_wav)


def _probe_video(video_path: str, log: logging.Logger) -> str:
    """Get basic video info using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                "-show_format",
                video_path,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            info_parts = []

            # Format info
            fmt = data.get("format", {})
            duration = float(fmt.get("duration", 0))
            if duration:
                info_parts.append(f"duration={_format_duration(duration)}")

            # Stream info
            for stream in data.get("streams", []):
                codec_type = stream.get("codec_type", "")
                codec_name = stream.get("codec_name", "?")
                if codec_type == "video":
                    w = stream.get("width", "?")
                    h = stream.get("height", "?")
                    info_parts.append(f"video={codec_name} {w}x{h}")
                elif codec_type == "audio":
                    sr = stream.get("sample_rate", "?")
                    ch = stream.get("channels", "?")
                    info_parts.append(f"audio={codec_name} {sr}Hz {ch}ch")

            return ", ".join(info_parts) if info_parts else "ok"
    except Exception as e:
        log.debug(f"ffprobe failed: {e}")
    return ""


def _get_audio_duration(wav_path: str, log: logging.Logger) -> float:
    """Get duration of WAV file in seconds."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                wav_path,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            return float(data.get("format", {}).get("duration", 0))
    except Exception:
        pass
    return 0.0


def _format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
