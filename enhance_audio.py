"""
# file: enhance_audio.py
Audio enhancement: noise reduction, dereverberation, speech enhancement.

Strategy (fallback chain):
  1. Try facebook/denoiser (DNS48 model) - best quality
  2. Try SpeechBrain MetricGAN+ - strong enhancement
  3. Try noisereduce + spectral cleanup - lightweight
  4. Last resort: basic bandpass filter only
"""

import logging
import time
from pathlib import Path
from typing import Optional


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

    log.warning("All enhancement methods failed. Applying normalization only.")
    _normalize_only(str(input_path), str(output_path), log)
    return str(output_path)


def _enhance_denoiser(input_wav: str, output_wav: str, log: logging.Logger):
    """Use facebook/denoiser DNS48 model."""
    import torch
    import torchaudio
    from denoiser import pretrained
    from denoiser.dsp import convert_audio

    log.info("Loading denoiser DNS48 model...")
    start = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = pretrained.dns64() if _check_dns64() else pretrained.dns48()
    model.to(device).eval()

    wav, sr = torchaudio.load(input_wav)
    wav = convert_audio(wav, sr, model.sample_rate, model.chin)
    wav = wav.to(device)

    log.info(f"Running denoiser (device={device})...")
    with torch.no_grad():
        enhanced = model(wav[None])[0]

    enhanced = enhanced.cpu()
    if enhanced.dim() == 1:
        enhanced = enhanced.unsqueeze(0)

    if model.sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(model.sample_rate, 16000)
        enhanced = resampler(enhanced)

    peak = enhanced.abs().max()
    if peak > 0:
        enhanced = enhanced / peak * 0.95

    torchaudio.save(output_wav, enhanced, 16000)
    log.info(f"Denoiser done in {time.time()-start:.1f}s")


def _check_dns64():
    """Check if dns64 is available (requires more memory)."""
    try:
        from denoiser import pretrained
        return hasattr(pretrained, "dns64")
    except Exception:
        return False


def _enhance_speechbrain(input_wav: str, output_wav: str, log: logging.Logger):
    """Use SpeechBrain MetricGAN+ enhancement."""
    import torch
    from speechbrain.inference.enhancement import SpectralMaskEnhancement

    log.info("Loading SpeechBrain MetricGAN+ model...")
    start = time.time()

    cache_dir = Path.home() / ".cache" / "speechbrain"
    cache_dir.mkdir(parents=True, exist_ok=True)

    model = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir=str(cache_dir / "metricgan-plus"),
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )

    log.info("Enhancing with MetricGAN+...")
    model.enhance_file(input_wav, output_wav)
    log.info(f"SpeechBrain done in {time.time()-start:.1f}s")


def _enhance_noisereduce(input_wav: str, output_wav: str, log: logging.Logger):
    """Use noisereduce plus spectral cleanup with speech-preserving safeguards."""
    import numpy as np
    import noisereduce as nr
    import scipy.io.wavfile as wav_io

    log.info("Loading audio for noisereduce...")
    start = time.time()

    sr, data = wav_io.read(input_wav)
    audio = _to_float_mono(data)

    log.info("Running adaptive noise reduction...")
    noise_clip = _select_noise_clip(audio, sr)

    if noise_clip is not None:
        audio_denoised = nr.reduce_noise(
            y=audio,
            sr=sr,
            y_noise=noise_clip,
            stationary=True,
            prop_decrease=0.72,
            n_fft=2048,
            win_length=2048,
            hop_length=512,
            n_std_thresh_stationary=1.4,
        )
    else:
        log.info("No reliable noise-only region found; skipping stationary denoise pass.")
        audio_denoised = audio

    audio_denoised = nr.reduce_noise(
        y=audio_denoised,
        sr=sr,
        stationary=False,
        prop_decrease=0.50,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
    )

    audio_denoised = _spectral_gate(audio_denoised, sr, noise_clip=noise_clip)
    audio_denoised = _bandpass_filter(audio_denoised, sr, low=70.0, high=min(7600.0, sr / 2.0 - 80.0))
    audio_denoised = _blend_with_original(audio, audio_denoised)
    audio_denoised = _upward_compress(audio_denoised, sr)
    audio_denoised = _quality_guard(audio, audio_denoised, log)
    audio_denoised = _normalize_audio(audio_denoised)

    wav_io.write(output_wav, sr, (audio_denoised * 32767).astype(np.int16))
    log.info(f"noisereduce done in {time.time()-start:.1f}s")


def _enhance_scipy(input_wav: str, output_wav: str, log: logging.Logger):
    """Use scipy-only cleanup with conservative speech preservation."""
    import numpy as np
    import scipy.io.wavfile as wav_io

    log.info("Running scipy spectral subtraction...")
    start = time.time()

    sr, data = wav_io.read(input_wav)
    audio = _to_float_mono(data)
    noise_clip = _select_noise_clip(audio, sr)

    audio_enhanced = _spectral_subtraction(audio, sr, noise_clip=noise_clip)
    audio_enhanced = _spectral_gate(audio_enhanced, sr, noise_clip=noise_clip)
    audio_enhanced = _bandpass_filter(audio_enhanced, sr, low=70.0, high=min(7600.0, sr / 2.0 - 80.0))
    audio_enhanced = _blend_with_original(audio, audio_enhanced)
    audio_enhanced = _upward_compress(audio_enhanced, sr)
    audio_enhanced = _quality_guard(audio, audio_enhanced, log)
    audio_enhanced = _normalize_audio(audio_enhanced)

    wav_io.write(output_wav, sr, (audio_enhanced * 32767).astype(np.int16))
    log.info(f"scipy spectral subtraction done in {time.time()-start:.1f}s")


def _spectral_subtraction(
    audio: "np.ndarray",
    sr: int,
    noise_clip: Optional["np.ndarray"] = None,
) -> "np.ndarray":
    """Simple spectral subtraction noise reduction."""
    import numpy as np
    import scipy.signal

    frame_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)

    _, _, zxx = scipy.signal.stft(
        audio,
        fs=sr,
        window="hann",
        nperseg=frame_len,
        noverlap=frame_len - hop_len,
        boundary=None,
        padded=False,
    )

    magnitude = np.abs(zxx)
    phase = np.angle(zxx)

    if noise_clip is not None and len(noise_clip) >= frame_len:
        _, _, noise_zxx = scipy.signal.stft(
            noise_clip,
            fs=sr,
            window="hann",
            nperseg=frame_len,
            noverlap=frame_len - hop_len,
            boundary=None,
            padded=False,
        )
        noise_estimate = np.mean(np.abs(noise_zxx), axis=1, keepdims=True)
    else:
        n_noise_frames = max(1, int(0.3 / (hop_len / sr)))
        noise_estimate = np.mean(magnitude[:, :n_noise_frames], axis=1, keepdims=True)

    noise_estimate = np.maximum(noise_estimate, 1e-10)
    enhanced_magnitude = np.maximum(magnitude - 1.7 * noise_estimate, 0.06 * magnitude)

    zxx_enhanced = enhanced_magnitude * np.exp(1j * phase)
    _, reconstructed = scipy.signal.istft(
        zxx_enhanced,
        fs=sr,
        window="hann",
        nperseg=frame_len,
        noverlap=frame_len - hop_len,
        input_onesided=True,
    )

    return _match_length(reconstructed.astype(np.float32), len(audio))


def _spectral_gate(
    audio: "np.ndarray",
    sr: int,
    noise_clip: Optional["np.ndarray"] = None,
) -> "np.ndarray":
    import numpy as np
    import scipy.signal

    frame_len = min(1024, max(256, int(0.032 * sr)))
    hop_len = max(128, int(0.008 * sr))

    _, _, zxx = scipy.signal.stft(
        audio,
        fs=sr,
        window="hann",
        nperseg=frame_len,
        noverlap=frame_len - hop_len,
        boundary=None,
        padded=False,
    )

    magnitude = np.abs(zxx)
    phase = np.angle(zxx)

    if noise_clip is not None and len(noise_clip) >= frame_len:
        _, _, noise_zxx = scipy.signal.stft(
            noise_clip,
            fs=sr,
            window="hann",
            nperseg=frame_len,
            noverlap=frame_len - hop_len,
            boundary=None,
            padded=False,
        )
        noise_profile = np.mean(np.abs(noise_zxx), axis=1, keepdims=True)
    else:
        noise_profile = np.percentile(magnitude, 20, axis=1, keepdims=True)

    mask = (magnitude - 0.9 * noise_profile) / (magnitude + 1e-8)
    mask = np.clip(mask, 0.18, 1.0)
    enhanced_magnitude = magnitude * (0.55 + 0.45 * mask)

    zxx_enhanced = enhanced_magnitude * np.exp(1j * phase)
    _, reconstructed = scipy.signal.istft(
        zxx_enhanced,
        fs=sr,
        window="hann",
        nperseg=frame_len,
        noverlap=frame_len - hop_len,
        input_onesided=True,
    )

    return _match_length(reconstructed.astype(np.float32), len(audio))


def _bandpass_filter(
    audio: "np.ndarray",
    sr: int,
    low: float = 80.0,
    high: float = 8000.0,
) -> "np.ndarray":
    """Apply bandpass filter for speech frequencies."""
    import scipy.signal

    nyq = sr / 2.0
    low_norm = low / nyq
    high_norm = min(high / nyq, 0.99)

    if low_norm <= 0 or high_norm >= 1 or low_norm >= high_norm:
        return audio.astype("float32")

    try:
        b, a = scipy.signal.butter(4, [low_norm, high_norm], btype="bandpass")
        return scipy.signal.filtfilt(b, a, audio).astype("float32")
    except Exception:
        return audio.astype("float32")


def _normalize_only(input_wav: str, output_wav: str, log: logging.Logger):
    """Last resort: just normalize the audio level."""
    import numpy as np
    import scipy.io.wavfile as wav_io

    sr, data = wav_io.read(input_wav)
    audio = _to_float_mono(data)
    audio = _bandpass_filter(audio, sr)
    audio = _normalize_audio(audio)

    wav_io.write(output_wav, sr, (audio * 32767).astype(np.int16))
    log.info("Normalization complete.")


def _select_noise_clip(audio: "np.ndarray", sr: int) -> Optional["np.ndarray"]:
    import numpy as np

    if len(audio) < max(400, int(sr * 0.25)):
        return None

    window = min(int(sr * 0.5), len(audio))
    hop = max(window // 2, 1)
    best_start = None
    best_rms = None

    for start in range(0, len(audio) - window + 1, hop):
        chunk = audio[start:start + window]
        rms = float(np.sqrt(np.mean(chunk ** 2) + 1e-12))
        if best_rms is None or rms < best_rms:
            best_rms = rms
            best_start = start

    if best_start is None or best_rms is None:
        return None

    global_rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
    if best_rms >= global_rms * 0.88:
        return None

    return audio[best_start:best_start + window]


def _blend_with_original(original: "np.ndarray", enhanced: "np.ndarray") -> "np.ndarray":
    import numpy as np

    original = _match_length(original, len(enhanced))
    enhanced = _match_length(enhanced, len(original))
    return (0.76 * enhanced + 0.24 * original).astype(np.float32)


def _upward_compress(audio: "np.ndarray", sr: int) -> "np.ndarray":
    import numpy as np

    window = max(64, int(0.05 * sr))
    kernel = np.ones(window, dtype=np.float32) / window
    envelope = np.sqrt(np.convolve(audio ** 2, kernel, mode="same") + 1e-8)
    gain = np.clip((0.08 / (envelope + 1e-4)) ** 0.22, 1.0, 1.45)
    return np.tanh(audio * gain * 1.05).astype(np.float32)


def _quality_guard(original: "np.ndarray", enhanced: "np.ndarray", log: logging.Logger) -> "np.ndarray":
    import numpy as np

    n = min(len(original), len(enhanced))
    if n == 0:
        return enhanced.astype(np.float32)

    original = original[:n].astype(np.float32)
    enhanced = enhanced[:n].astype(np.float32)

    orig_rms = float(np.sqrt(np.mean(original ** 2) + 1e-12))
    enh_rms = float(np.sqrt(np.mean(enhanced ** 2) + 1e-12))
    rms_ratio = enh_rms / max(orig_rms, 1e-6)

    orig_std = float(np.std(original))
    enh_std = float(np.std(enhanced))
    if orig_std > 1e-6 and enh_std > 1e-6:
        corr = float(np.corrcoef(original, enhanced)[0, 1])
    else:
        corr = 1.0

    if not np.isfinite(corr):
        corr = 1.0

    if corr < 0.55 or rms_ratio < 0.35 or rms_ratio > 1.9:
        log.info("Enhancement looked destructive; falling back to a conservative original/enhanced mix.")
        return (0.55 * original + 0.45 * enhanced).astype(np.float32)

    if corr < 0.72:
        log.info("Enhancement changed speech strongly; using a safer blend with the original audio.")
        return (0.38 * original + 0.62 * enhanced).astype(np.float32)

    return enhanced.astype(np.float32)


def _normalize_audio(audio: "np.ndarray") -> "np.ndarray":
    import numpy as np

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0:
        audio = audio / peak * 0.95
    return audio.astype(np.float32)


def _to_float_mono(data: "np.ndarray") -> "np.ndarray":
    import numpy as np

    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    else:
        audio = data.astype(np.float32)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    return audio.astype(np.float32)


def _match_length(audio: "np.ndarray", target_len: int) -> "np.ndarray":
    import numpy as np

    if len(audio) == target_len:
        return audio.astype(np.float32)
    if len(audio) > target_len:
        return audio[:target_len].astype(np.float32)
    return np.pad(audio, (0, target_len - len(audio))).astype(np.float32)
