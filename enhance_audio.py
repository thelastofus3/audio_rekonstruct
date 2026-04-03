"""
# file: enhance_audio.py
Audio enhancement: noise reduction, dereverberation, speech enhancement.

Strategy (fallback chain):
  1. Try facebook/denoiser (DNS48 model) — best quality
  2. Try SpeechBrain MetricGAN+ — strong enhancement
  3. Try noisereduce + scipy spectral subtraction — lightweight
  4. Last resort: basic bandpass filter only
"""

import logging
import os
import sys
import time
from pathlib import Path


def enhance_audio(
    input_wav: str,
    output_wav: str,
    logger: logging.Logger = None,
) -> str:
    """
    Enhance speech audio using best available method.

    Args:
        input_wav: Path to input WAV (16kHz mono recommended).
        output_wav: Path to save enhanced WAV.
        logger: Optional logger.

    Returns:
        Path to enhanced WAV file.
    """
    log = logger or logging.getLogger(__name__)

    input_path = Path(input_wav)
    output_path = Path(output_wav)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input WAV not found: {input_wav}")

    log.info("Attempting enhancement methods...")

    # Try each method in order
    methods = [
        ("facebook/denoiser", _enhance_denoiser),
        ("SpeechBrain MetricGAN+", _enhance_speechbrain),
        ("noisereduce", _enhance_noisereduce),
        ("scipy spectral subtraction", _enhance_scipy),
    ]

    for method_name, method_fn in methods:
        log.info(f"Trying {method_name}...")
        try:
            method_fn(str(input_path), str(output_path), log)
            if output_path.exists() and output_path.stat().st_size > 0:
                log.info(f"Enhancement successful using: {method_name}")
                return str(output_path)
        except ImportError as e:
            log.info(f"{method_name} not available: {e}")
        except Exception as e:
            log.warning(f"{method_name} failed: {e}")

    # Final fallback: copy with normalization only
    log.warning("All enhancement methods failed. Applying normalization only.")
    _normalize_only(str(input_path), str(output_path), log)
    return str(output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Method 1: facebook/denoiser
# ─────────────────────────────────────────────────────────────────────────────

def _enhance_denoiser(input_wav: str, output_wav: str, log: logging.Logger):
    """Use facebook/denoiser DNS48 model."""
    import torch
    from denoiser import pretrained
    from denoiser.dsp import convert_audio
    import torchaudio

    log.info("Loading denoiser DNS48 model...")
    start = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = pretrained.dns64() if _check_dns64() else pretrained.dns48()
    model.to(device).eval()

    # Load audio
    wav, sr = torchaudio.load(input_wav)
    wav = convert_audio(wav, sr, model.sample_rate, model.chin)
    wav = wav.to(device)

    log.info(f"Running denoiser (device={device})...")
    with torch.no_grad():
        enhanced = model(wav[None])[0]

    # Save
    enhanced = enhanced.cpu()
    if enhanced.dim() == 1:
        enhanced = enhanced.unsqueeze(0)

    # Resample to 16kHz if needed
    if model.sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(model.sample_rate, 16000)
        enhanced = resampler(enhanced)

    # Normalize
    peak = enhanced.abs().max()
    if peak > 0:
        enhanced = enhanced / peak * 0.95

    torchaudio.save(output_wav, enhanced, 16000)
    log.info(f"Denoiser done in {time.time()-start:.1f}s")


def _check_dns64():
    """Check if dns64 is available (requires more memory)."""
    try:
        from denoiser import pretrained
        return hasattr(pretrained, 'dns64')
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Method 2: SpeechBrain MetricGAN+
# ─────────────────────────────────────────────────────────────────────────────

def _enhance_speechbrain(input_wav: str, output_wav: str, log: logging.Logger):
    """Use SpeechBrain MetricGAN+ enhancement."""
    import torch
    import torchaudio
    from speechbrain.inference.enhancement import SpectralMaskEnhancement

    log.info("Loading SpeechBrain MetricGAN+ model...")
    start = time.time()

    # Save model to cache
    cache_dir = Path.home() / ".cache" / "speechbrain"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir=str(cache_dir / "metricgan-plus"),
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )

    log.info("Enhancing with MetricGAN+...")
    enhanced = model.enhance_file(input_wav, output_wav)
    log.info(f"SpeechBrain done in {time.time()-start:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# Method 3: noisereduce
# ─────────────────────────────────────────────────────────────────────────────

def _enhance_noisereduce(input_wav: str, output_wav: str, log: logging.Logger):
    """Use noisereduce library with scipy for spectral subtraction."""
    import numpy as np
    import noisereduce as nr
    import scipy.io.wavfile as wav_io
    import scipy.signal

    log.info("Loading audio for noisereduce...")
    start = time.time()

    sr, data = wav_io.read(input_wav)

    # Convert to float32
    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    else:
        audio = data.astype(np.float32)

    # Mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    log.info("Running noise reduction (stationary + non-stationary)...")

    # First pass: stationary noise reduction (use first 0.5s as noise profile)
    noise_sample_len = min(int(sr * 0.5), len(audio) // 4)
    noise_clip = audio[:noise_sample_len] if noise_sample_len > 100 else audio

    audio_denoised = nr.reduce_noise(
        y=audio,
        sr=sr,
        y_noise=noise_clip,
        stationary=True,
        prop_decrease=0.85,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_std_thresh_stationary=1.5,
    )

    # Second pass: non-stationary noise reduction
    audio_denoised2 = nr.reduce_noise(
        y=audio_denoised,
        sr=sr,
        stationary=False,
        prop_decrease=0.75,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
    )

    # Apply bandpass filter (80Hz - 8000Hz for speech)
    audio_filtered = _bandpass_filter(audio_denoised2, sr, low=80.0, high=8000.0)

    # Normalize
    peak = np.abs(audio_filtered).max()
    if peak > 0:
        audio_filtered = audio_filtered / peak * 0.95

    # Save as 16-bit PCM
    audio_int16 = (audio_filtered * 32767).astype(np.int16)
    wav_io.write(output_wav, sr, audio_int16)

    log.info(f"noisereduce done in {time.time()-start:.1f}s")


# ─────────────────────────────────────────────────────────────────────────────
# Method 4: scipy spectral subtraction only
# ─────────────────────────────────────────────────────────────────────────────

def _enhance_scipy(input_wav: str, output_wav: str, log: logging.Logger):
    """Basic spectral subtraction using scipy only."""
    import numpy as np
    import scipy.io.wavfile as wav_io
    import scipy.signal

    log.info("Running scipy spectral subtraction...")
    start = time.time()

    sr, data = wav_io.read(input_wav)

    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    else:
        audio = data.astype(np.float32)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Spectral subtraction
    audio_enhanced = _spectral_subtraction(audio, sr)

    # Bandpass filter
    audio_filtered = _bandpass_filter(audio_enhanced, sr, low=80.0, high=8000.0)

    # Normalize
    peak = np.abs(audio_filtered).max()
    if peak > 0:
        audio_filtered = audio_filtered / peak * 0.95

    audio_int16 = (audio_filtered * 32767).astype(np.int16)
    wav_io.write(output_wav, sr, audio_int16)

    log.info(f"scipy spectral subtraction done in {time.time()-start:.1f}s")


def _spectral_subtraction(audio: "np.ndarray", sr: int) -> "np.ndarray":
    """Simple spectral subtraction noise reduction."""
    import numpy as np
    import scipy.signal

    frame_len = int(0.025 * sr)   # 25ms frames
    hop_len = int(0.010 * sr)     # 10ms hop

    # Use STFT
    f, t, Zxx = scipy.signal.stft(
        audio,
        fs=sr,
        window="hann",
        nperseg=frame_len,
        noverlap=frame_len - hop_len,
    )

    magnitude = np.abs(Zxx)
    phase = np.angle(Zxx)

    # Estimate noise from first 0.3s (assumed to be noise)
    n_noise_frames = max(1, int(0.3 / (hop_len / sr)))
    noise_estimate = np.mean(magnitude[:, :n_noise_frames], axis=1, keepdims=True)
    noise_estimate = np.maximum(noise_estimate, 1e-10)

    # Over-subtraction with flooring
    alpha = 2.0   # over-subtraction factor
    beta = 0.01   # spectral floor

    enhanced_magnitude = magnitude - alpha * noise_estimate
    enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)

    # Reconstruct
    Zxx_enhanced = enhanced_magnitude * np.exp(1j * phase)
    _, reconstructed = scipy.signal.istft(
        Zxx_enhanced,
        fs=sr,
        window="hann",
        nperseg=frame_len,
        noverlap=frame_len - hop_len,
    )

    # Align length
    min_len = min(len(audio), len(reconstructed))
    return reconstructed[:min_len].astype(np.float32)


def _bandpass_filter(
    audio: "np.ndarray",
    sr: int,
    low: float = 80.0,
    high: float = 8000.0,
) -> "np.ndarray":
    """Apply bandpass filter for speech frequencies."""
    import scipy.signal
    import numpy as np

    nyq = sr / 2.0
    low_norm = low / nyq
    high_norm = min(high / nyq, 0.99)

    if low_norm <= 0 or high_norm >= 1 or low_norm >= high_norm:
        return audio

    try:
        b, a = scipy.signal.butter(4, [low_norm, high_norm], btype="bandpass")
        return scipy.signal.filtfilt(b, a, audio).astype(np.float32)
    except Exception:
        return audio


def _normalize_only(input_wav: str, output_wav: str, log: logging.Logger):
    """Last resort: just normalize the audio level."""
    import numpy as np
    import scipy.io.wavfile as wav_io

    sr, data = wav_io.read(input_wav)

    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    else:
        audio = data.astype(np.float32)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Apply bandpass
    audio = _bandpass_filter(audio, sr)

    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.95

    audio_int16 = (audio * 32767).astype(np.int16)
    wav_io.write(output_wav, sr, audio_int16)
    log.info("Normalization complete.")
