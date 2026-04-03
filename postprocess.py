"""
# file: postprocess.py
Transcript postprocessing:
  - Mark unclear/inaudible segments with [неразборчиво], [предположение], [вариант1/вариант2]
  - LLM restoration of unclear text via internal Claude API (no external API key package needed)
  - Optional TTS audio restoration: synthesize restored text and splice back into audio
  - Build clean full-text transcript

LLM provider options:
  "claude"    — uses api.anthropic.com directly via requests (requires ANTHROPIC_API_KEY)
  "openai"    — uses OpenAI API (requires OPENAI_API_KEY)
  "anthropic" — same as "claude" but via anthropic package
  "local"     — Ollama or local OpenAI-compatible endpoint
"""

import json
import logging
import os
import re
import time
from typing import List, Optional, Tuple


INAUDIBLE_TAG = "[неразборчиво]"
ASSUMPTION_TAG = "[предположение]"


def postprocess_transcript(
        segments: List[dict],
        use_llm: bool = False,
        llm_provider: str = "claude",
        confidence_threshold: float = 0.5,
        language: str = "auto",
        logger: logging.Logger = None,
        # TTS audio restoration options
        restore_audio: bool = False,
        audio_path: str = None,
        output_audio_path: str = None,
        tts_language: str = "ru",
) -> Tuple[List[dict], str]:
    """
    Postprocess ASR segments: mark unclear parts, optionally use LLM + TTS.

    Args:
        segments: List of ASR segment dicts.
        use_llm: Whether to use LLM for unclear text restoration.
        llm_provider: 'claude', 'openai', 'anthropic', or 'local'.
        confidence_threshold: Threshold below which segment is unclear.
        language: Language code for LLM prompting.
        logger: Optional logger.
        restore_audio: If True, synthesize restored text via TTS and splice into audio.
        audio_path: Path to source WAV (required if restore_audio=True).
        output_audio_path: Path to save restored audio WAV.
        tts_language: Language for TTS synthesis (e.g. 'ru', 'en').

    Returns:
        (processed_segments, full_text)
    """
    log = logger or logging.getLogger(__name__)

    if not segments:
        log.warning("No segments to postprocess.")
        return [], ""

    log.info(f"Postprocessing {len(segments)} segments...")

    # Step 1: Mark unclear segments
    processed = _mark_unclear_segments(segments, confidence_threshold, log)

    # Step 2: Fix word-level unclear markers
    processed = _mark_unclear_words(processed, confidence_threshold=0.3, log=log)

    # Step 3: LLM restoration (optional)
    if use_llm:
        _wre = re.compile(r'\[[^\]]+\?\]')
        unclear_count = sum(
            1 for s in processed
            if s.get("is_unclear") or _wre.search(s.get("text", ""))
        )
        if unclear_count > 0:
            log.info(f"Running LLM restoration on {unclear_count} unclear segments...")
            processed = _llm_restore(processed, llm_provider, language, log)
        else:
            log.info("No unclear segments for LLM restoration.")

    # Step 4: TTS audio restoration — splice synthesized speech into audio file
    if restore_audio and audio_path and output_audio_path:
        _unclear_re = re.compile(r'\[[^\]]+\?\]')
        tts_candidates = [
            s for s in processed
            if s.get("restoration_method", "").startswith("llm_")
               or ASSUMPTION_TAG in s.get("text", "")
               or bool(_unclear_re.search(s.get("text", "")))
        ]
        if tts_candidates:
            log.info(
                f"\n[TTS AUDIO RESTORATION] Synthesizing and splicing "
                f"{len(tts_candidates)} segment(s) into audio..."
            )
            _restore_audio_tts(processed, audio_path, output_audio_path, tts_language, log)
        else:
            log.info("No unclear segments found — audio unchanged.")
            import shutil
            shutil.copy(audio_path, output_audio_path)

    # Step 5: Build full text
    full_text = _build_full_text(processed)

    # Step 6: Statistics
    total = len(processed)
    unclear_count = sum(1 for s in processed if s.get("is_unclear"))
    avg_conf = sum(s.get("confidence", 0) for s in processed) / max(total, 1)

    log.info(
        f"Postprocessing complete: {total} segments, "
        f"{unclear_count} unclear ({unclear_count/max(total,1)*100:.0f}%), "
        f"avg confidence={avg_conf:.2f}"
    )

    return processed, full_text


# ─────────────────────────────────────────────────────────────────────────────
# Marking unclear segments
# ─────────────────────────────────────────────────────────────────────────────

def _mark_unclear_segments(
        segments: List[dict],
        threshold: float,
        log: logging.Logger,
) -> List[dict]:
    """Mark segments as unclear based on confidence and heuristics."""
    result = []

    for seg in segments:
        seg = dict(seg)  # copy
        text = seg.get("text", "").strip()
        confidence = seg.get("confidence", 1.0)
        no_speech_prob = seg.get("no_speech_prob", 0.0)

        # Already marked unclear by ASR
        if seg.get("is_unclear", False):
            if not text or _is_noise_text(text):
                seg["text"] = INAUDIBLE_TAG
                seg["restoration_tag"] = "inaudible"
            elif confidence < threshold * 0.6:
                seg["text"] = INAUDIBLE_TAG
                seg["restoration_tag"] = "inaudible"
            else:
                clean_text = _clean_noise_artifacts(text)
                seg["text"] = f"{ASSUMPTION_TAG} {clean_text}"
                seg["restoration_tag"] = "assumption"

        elif _is_noise_text(text):
            seg["text"] = INAUDIBLE_TAG
            seg["is_unclear"] = True
            seg["restoration_tag"] = "noise"

        elif confidence < threshold:
            clean_text = _clean_noise_artifacts(text)
            if clean_text:
                seg["text"] = f"{ASSUMPTION_TAG} {clean_text}"
                seg["restoration_tag"] = "low_confidence"
            else:
                seg["text"] = INAUDIBLE_TAG
                seg["restoration_tag"] = "inaudible"
            seg["is_unclear"] = True

        else:
            seg["text"] = _clean_noise_artifacts(text)
            seg["is_unclear"] = False
            seg["restoration_tag"] = None

        result.append(seg)

    return result


def _mark_unclear_words(
        segments: List[dict],
        confidence_threshold: float = 0.3,
        log: logging.Logger = None,
) -> List[dict]:
    """Mark individual low-confidence words within segments."""
    result = []

    for seg in segments:
        seg = dict(seg)
        words = seg.get("words", [])

        if not words or seg.get("restoration_tag") in ("inaudible", "noise"):
            result.append(seg)
            continue

        if INAUDIBLE_TAG in seg.get("text", "") or ASSUMPTION_TAG in seg.get("text", ""):
            result.append(seg)
            continue

        new_words = []
        has_unclear_words = False

        for w in words:
            word = w.get("word", "")
            word_conf = w.get("confidence", 1.0)

            if word_conf < confidence_threshold and word.strip():
                marked = w.copy()
                marked["is_unclear"] = True
                new_words.append(marked)
                has_unclear_words = True
            else:
                new_words.append(w)

        if has_unclear_words:
            text_parts = []
            for w in new_words:
                word_text = w.get("word", "")
                if w.get("is_unclear") and word_text.strip():
                    text_parts.append(f"[{word_text.strip()}?]")
                else:
                    text_parts.append(word_text)
            seg["text"] = "".join(text_parts).strip()
            seg["words"] = new_words

        result.append(seg)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# LLM restoration
# ─────────────────────────────────────────────────────────────────────────────

def _llm_restore(
        segments: List[dict],
        provider: str,
        language: str,
        log: logging.Logger,
) -> List[dict]:
    """Use LLM to restore unclear segments using surrounding context."""

    llm_fn = _get_llm_function(provider, log)
    if not llm_fn:
        log.warning("LLM not available, skipping restoration.")
        return segments

    result = list(segments)

    # Regex to detect word-level unclear markers like [Listen,?] or [word?]
    _unclear_word_re = re.compile(r'\[[^\]]+\?\]')

    for i, seg in enumerate(result):
        seg_text = seg.get("text", "")

        # Process segment if:
        #   a) flagged as unclear by ASR confidence, OR
        #   b) contains word-level markers like [word?] even if overall confidence was OK
        has_unclear_words = bool(_unclear_word_re.search(seg_text))
        if not seg.get("is_unclear") and not has_unclear_words:
            continue

        # If segment has word-level markers, treat as assumption (not fully inaudible)
        if has_unclear_words and not seg.get("is_unclear"):
            seg = dict(seg)
            seg["is_unclear"] = True
            result[i] = seg

        if seg.get("text") == INAUDIBLE_TAG:
            context_before = _get_context(result, i, window=3, before=True)
            context_after = _get_context(result, i, window=2, before=False)

            if not context_before and not context_after:
                continue

            prompt = _build_restoration_prompt(
                unclear_text="",
                context_before=context_before,
                context_after=context_after,
                language=language,
                is_completely_inaudible=True,
            )

            try:
                restored = llm_fn(prompt)
                if restored and len(restored) > 2:
                    result[i] = dict(seg)
                    result[i]["text"] = f"[предположение] {restored}"
                    result[i]["restoration_method"] = f"llm_{provider}"
                    log.debug(f"LLM restored segment {i}: '{restored[:50]}...'")
            except Exception as e:
                log.warning(f"LLM restoration failed for segment {i}: {e}")

        elif ASSUMPTION_TAG in seg.get("text", "") or "[" in seg.get("text", ""):
            raw_text = seg.get("text", "").replace(ASSUMPTION_TAG, "").strip()
            context_before = _get_context(result, i, window=3, before=True)
            context_after = _get_context(result, i, window=2, before=False)

            prompt = _build_restoration_prompt(
                unclear_text=raw_text,
                context_before=context_before,
                context_after=context_after,
                language=language,
                is_completely_inaudible=False,
            )

            try:
                restored = llm_fn(prompt)
                if restored and len(restored) > 2:
                    result[i] = dict(seg)
                    result[i]["text"] = f"[предположение] {restored}"
                    result[i]["restoration_method"] = f"llm_{provider}"
                    log.debug(f"LLM clarified segment {i}: '{restored[:50]}...'")
            except Exception as e:
                log.warning(f"LLM clarification failed for segment {i}: {e}")

    return result


def _get_llm_function(provider: str, log: logging.Logger):
    """
    Get LLM callable based on provider.

    Providers:
      "local"     — Ollama running locally (FREE, no key needed) — DEFAULT
      "openai"    — OpenAI API (requires OPENAI_API_KEY)
      "claude"    — Anthropic Claude API (requires ANTHROPIC_API_KEY)
    """

    if provider == "local":
        return _get_local_llm_fn(log)

    elif provider in ("claude", "anthropic"):
        import requests as _requests
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            log.warning("ANTHROPIC_API_KEY not set. Falling back to local LLM.")
            return _get_local_llm_fn(log)

        def call_claude(prompt: str) -> str:
            response = _requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 200,
                    "system": (
                        "You are a professional transcription editor. "
                        "Restore unclear speech based on context. "
                        "Return ONLY the restored text, no explanations, no preamble."
                    ),
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            for block in data.get("content", []):
                if block.get("type") == "text":
                    return block["text"].strip()
            return ""

        return call_claude

    elif provider == "openai":
        try:
            import openai
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                log.warning("OPENAI_API_KEY not set. Falling back to local LLM.")
                return _get_local_llm_fn(log)

            client = openai.OpenAI(api_key=api_key)

            def call_openai(prompt: str) -> str:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": (
                            "You are a professional transcription editor. "
                            "Restore unclear speech based on context. "
                            "Return ONLY the restored text, no explanations."
                        )},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=200,
                    temperature=0.3,
                )
                return response.choices[0].message.content.strip()

            return call_openai
        except ImportError:
            log.warning("openai package not installed. Falling back to local LLM.")
            return _get_local_llm_fn(log)

    return _get_local_llm_fn(log)


def _get_local_llm_fn(log: logging.Logger):
    """
    Connect to a locally running LLM via Ollama (recommended) or any
    OpenAI-compatible server (LM Studio, llama.cpp, etc.).

    Setup (one-time):
      1. Install Ollama: https://ollama.com
      2. Pull a model:
           ollama pull gemma3          # good for Russian (~5 GB)
           ollama pull qwen2.5:7b      # excellent multilingual (~5 GB)
           ollama pull mistral         # good English (~4 GB)
      3. Ollama starts automatically — no extra commands needed.

    Custom server URL: set LOCAL_LLM_URL env var (default: http://localhost:11434/v1)
    Custom model:      set LOCAL_LLM_MODEL env var  (default: auto-detect from Ollama)
    """
    import requests as _requests

    base_url = os.environ.get("LOCAL_LLM_URL", "http://localhost:11434/v1")
    model = os.environ.get("LOCAL_LLM_MODEL", "")

    # Auto-detect available model from Ollama if model not specified
    if not model:
        model = _detect_ollama_model(base_url, log)
        if not model:
            log.warning(
                "No local LLM model found. "
                "Install Ollama (https://ollama.com) and run: ollama pull qwen2.5:7b"
            )
            return None

    log.info(f"Local LLM: {base_url}, model={model}")
    log.info("Note: first request may take 30-120s while model loads into memory.")

    # Verify Ollama is reachable via lightweight ping (no model inference)
    ollama_base = base_url.replace("/v1", "")
    try:
        ping = _requests.get(f"{ollama_base}/api/version", timeout=5)
        if ping.status_code != 200:
            raise ConnectionError(f"Ollama ping returned {ping.status_code}")
        log.info(f"Ollama is running (version: {ping.json().get('version', '?')})")
    except Exception as e:
        log.warning(f"Cannot reach Ollama at {ollama_base}: {e}")
        log.warning("Make sure Ollama is running. Start it with: ollama serve")
        return None

    # Timeout 300s — model cold-start on CPU can take 1-2 min
    # Override with: set LOCAL_LLM_TIMEOUT=600
    _TIMEOUT = int(os.environ.get("LOCAL_LLM_TIMEOUT", "300"))

    def call_local(prompt: str) -> str:
        resp = _requests.post(
            f"{base_url}/chat/completions",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a professional transcription editor. "
                            "Restore unclear speech based on context. "
                            "Return ONLY the restored text, no explanations, no preamble."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 200,
                "temperature": 0.3,
                "stream": False,
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    return call_local


def _detect_ollama_model(base_url: str, log: logging.Logger) -> str:
    """
    Ask Ollama which models are installed and pick the best one for transcription.
    Preferred order: qwen2.5, gemma3, mistral, llama3, phi3 — any available.
    """
    import requests as _requests

    # Ollama-specific models list endpoint
    ollama_base = base_url.replace("/v1", "")
    preferred = ["qwen2.5", "gemma3", "mistral", "llama3", "phi3", "llama2"]

    try:
        resp = _requests.get(f"{ollama_base}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            if models:
                log.info(f"Ollama installed models: {', '.join(models)}")
                # Pick by preference
                for pref in preferred:
                    for m in models:
                        if m.startswith(pref):
                            return m
                return models[0]  # just use whatever is installed
    except Exception:
        pass

    # Fallback: try OpenAI-compatible /models endpoint
    try:
        resp = _requests.get(f"{base_url}/models", timeout=5)
        if resp.status_code == 200:
            models = [m["id"] for m in resp.json().get("data", [])]
            if models:
                log.info(f"Available models: {', '.join(models)}")
                for pref in preferred:
                    for m in models:
                        if pref in m:
                            return m
                return models[0]
    except Exception:
        pass

    return ""



def _restore_audio_tts(
        segments: List[dict],
        source_audio_path: str,
        output_audio_path: str,
        tts_language: str,
        log: logging.Logger,
):
    """
    For each LLM-restored segment, synthesize speech via TTS and splice it
    back into the source audio at the correct timestamp.

    Requires at least one TTS backend:
      - gtts  (pip install gtts)   — free, needs internet, good quality
      - pyttsx3 (pip install pyttsx3) — offline, system voices

    Output is saved to output_audio_path.
    """
    import numpy as np
    import scipy.io.wavfile as wav_io
    from pathlib import Path
    import tempfile

    log.info(f"Loading source audio: {source_audio_path}")
    sr, data = wav_io.read(source_audio_path)

    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    else:
        audio = data.astype(np.float32)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    tts_fn = _get_tts_function(tts_language, log)
    if not tts_fn:
        log.warning("No TTS backend available. Skipping audio restoration.")
        return

    restored_count = 0

    _variant_re = re.compile(r'\[([^/\]]+)/[^\]]+\]')
    _unclear_word_re2 = re.compile(r'\[([^\]]+)\?\]')

    with tempfile.TemporaryDirectory() as tmpdir:
        for seg in segments:
            seg_text_raw = seg.get("text", "")

            # Process segment if LLM restored it OR if it still has unclear word markers
            was_llm_restored = seg.get("restoration_method", "").startswith("llm_")
            has_unclear_markers = (
                    INAUDIBLE_TAG in seg_text_raw
                    or ASSUMPTION_TAG in seg_text_raw
                    or bool(_unclear_word_re2.search(seg_text_raw))
            )
            if not was_llm_restored and not has_unclear_markers:
                continue

            # Build clean text for TTS
            text = seg_text_raw
            text = text.replace(ASSUMPTION_TAG, "").replace(INAUDIBLE_TAG, "").strip()
            # [вариантA/вариантB] → pick first variant
            text = _variant_re.sub(r'\1', text)
            # [слово?] → keep the word
            text = _unclear_word_re2.sub(r'\1', text)
            text = text.strip()

            if not text:
                log.info(f"  Segment {seg['id']} ({seg.get('start',0):.1f}s): fully inaudible, silence kept.")
                continue

            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", 0.0)
            seg_duration = seg_end - seg_start

            if seg_duration <= 0.05:
                continue

            # Synthesize to temp WAV
            tts_wav = os.path.join(tmpdir, f"tts_{seg['id']}.wav")
            try:
                tts_fn(text, tts_wav)
            except Exception as e:
                log.warning(f"TTS synthesis failed for segment {seg['id']}: {e}")
                continue

            if not os.path.exists(tts_wav) or os.path.getsize(tts_wav) == 0:
                continue

            # Load TTS audio and resample to match source SR
            tts_sr, tts_data = wav_io.read(tts_wav)
            if tts_data.dtype == np.int16:
                tts_audio = tts_data.astype(np.float32) / 32768.0
            else:
                tts_audio = tts_data.astype(np.float32)

            if tts_audio.ndim > 1:
                tts_audio = tts_audio.mean(axis=1)

            # Resample TTS to source SR if needed
            if tts_sr != sr:
                tts_audio = _resample(tts_audio, tts_sr, sr)

            tts_len = len(tts_audio)
            target_samples = int(seg_duration * sr)
            ratio = tts_len / max(target_samples, 1)

            log.info(
                f"  Segment {seg['id']} slot={seg_duration:.2f}s, "
                f"TTS={tts_len/sr:.2f}s, ratio={ratio:.2f}"
            )

            # ── Time-stretch TTS to fit the gap without changing pitch ────────
            # Strategy:
            #   ratio <= 1.25 : TTS fits or is slightly longer → pad/trim silently
            #   ratio  > 1.25 : TTS is much longer than slot →
            #       try librosa PSOLA (pitch-preserving), fallback to
            #       shifting the rest of the audio to make room (natural result)
            if ratio <= 1.25:
                # TTS fits within slot (or barely over) — pad with silence or trim
                if tts_len < target_samples:
                    # Pad end with silence to fill slot
                    tts_audio = np.pad(tts_audio, (0, target_samples - tts_len))
                else:
                    # Trim to slot (ratio ≤ 1.25, so at most ~100ms clipped)
                    tts_audio = tts_audio[:target_samples]
            else:
                # TTS is significantly longer than the original gap.
                # Try pitch-preserving stretch first (librosa), otherwise
                # just insert at natural length (shifts subsequent audio).
                stretched = _pitch_preserving_stretch(tts_audio, sr, target_samples, log)
                if stretched is not None:
                    tts_audio = stretched
                    log.info(f"  Used pitch-preserving stretch (ratio={ratio:.2f})")
                else:
                    # Insert at natural TTS length — audio after this point
                    # will be shifted right by (tts_len - target_samples) samples.
                    log.info(
                        f"  TTS longer than slot by {(tts_len-target_samples)/sr:.2f}s "
                        f"— inserting at natural length (audio shifted right)"
                    )
                    splice_start = int(seg_start * sr)
                    gap_end = int(seg_end * sr)
                    tail = audio[gap_end:].copy()
                    audio = np.concatenate([
                        audio[:splice_start],
                        tts_audio,
                        tail,
                    ])
                    restored_count += 1
                    seg_id = seg['id']
                    log.info(
                        f"  Spliced TTS for segment {seg_id} "
                        f"({seg_start:.1f}s): \"{text[:50]}\""
                    )
                    continue  # skip the normal splice below

            # Apply fade-in/out to avoid clicks (10ms)
            fade_samples = min(int(0.010 * sr), len(tts_audio) // 4)
            if fade_samples > 0:
                fade = np.linspace(0, 1, fade_samples)
                tts_audio[:fade_samples] *= fade
                tts_audio[-fade_samples:] *= fade[::-1]

            # Normalize TTS volume to match surrounding audio level
            start_sample = int(seg_start * sr)
            end_sample = min(int(seg_end * sr), len(audio))
            if end_sample > start_sample and end_sample <= len(audio):
                source_rms = np.sqrt(np.mean(audio[start_sample:end_sample] ** 2))
                tts_rms = np.sqrt(np.mean(tts_audio ** 2))
                if tts_rms > 1e-6 and source_rms > 1e-6:
                    tts_audio = tts_audio * (source_rms / tts_rms)

            # Splice into audio
            splice_start = int(seg_start * sr)
            splice_end = splice_start + len(tts_audio)

            if splice_end > len(audio):
                audio = np.pad(audio, (0, splice_end - len(audio)))

            audio[splice_start:splice_end] = tts_audio
            restored_count += 1
            log.info(
                f"  Spliced TTS for segment {seg['id']} "
                f"({seg_start:.1f}s-{seg_end:.1f}s): \"{text[:40]}\""
            )

    if restored_count == 0:
        log.warning("No segments were spliced (TTS synthesis may have failed for all).")
        return

    # Save output
    audio_out = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_out * 32767).astype(np.int16)
    wav_io.write(output_audio_path, sr, audio_int16)
    log.info(f"Restored audio saved: {output_audio_path} ({restored_count} segments spliced)")


def _get_tts_function(language: str, log: logging.Logger):
    """Return a TTS function: tts_fn(text, output_wav_path)."""

    # Try gTTS (Google TTS) — free, online, best quality
    try:
        from gtts import gTTS
        import subprocess

        def gtts_synthesize(text: str, output_wav: str):
            import tempfile, os
            # gTTS outputs MP3, convert to WAV via ffmpeg
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_mp3 = f.name
            try:
                tts = gTTS(text=text, lang=language, slow=False)
                tts.save(tmp_mp3)
                # Convert MP3 → WAV
                result = subprocess.run(
                    ["ffmpeg", "-y", "-i", tmp_mp3,
                     "-ar", "16000", "-ac", "1",
                     "-acodec", "pcm_s16le", output_wav],
                    capture_output=True, timeout=30,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"ffmpeg MP3→WAV failed: {result.stderr[-300:]}")
            finally:
                if os.path.exists(tmp_mp3):
                    os.unlink(tmp_mp3)

        log.info("TTS backend: gTTS (Google TTS)")
        return gtts_synthesize
    except ImportError:
        log.info("gTTS not available, trying pyttsx3...")

    # Try pyttsx3 — offline, system voices
    try:
        import pyttsx3
        import wave
        import tempfile

        engine = pyttsx3.init()
        # Try to set language voice
        voices = engine.getProperty("voices")
        for v in voices:
            if language.lower() in (v.languages or []) or language in v.id.lower():
                engine.setProperty("voice", v.id)
                break
        engine.setProperty("rate", 160)

        def pyttsx3_synthesize(text: str, output_wav: str):
            engine.save_to_file(text, output_wav)
            engine.runAndWait()

        log.info("TTS backend: pyttsx3 (offline)")
        return pyttsx3_synthesize
    except ImportError:
        log.warning("No TTS backend found. Install: pip install gtts  or  pip install pyttsx3")
        return None


def _resample(audio: "np.ndarray", from_sr: int, to_sr: int) -> "np.ndarray":
    """Resample audio array from from_sr to to_sr."""
    import numpy as np
    if from_sr == to_sr:
        return audio
    try:
        import scipy.signal
        n_samples = int(len(audio) * to_sr / from_sr)
        return scipy.signal.resample(audio, n_samples).astype(np.float32)
    except Exception:
        return audio


def _pitch_preserving_stretch(
        audio: "np.ndarray",
        sr: int,
        target_len: int,
        log: "logging.Logger",
) -> "Optional[np.ndarray]":
    """
    Stretch audio to target_len samples WITHOUT changing pitch.
    Uses librosa phase vocoder (PSOLA-like). Returns None if unavailable.
    Max compression ratio allowed: 0.5 (never more than 2x faster).
    """
    import numpy as np

    if len(audio) == target_len:
        return audio

    rate = len(audio) / max(target_len, 1)

    # Refuse extreme compression — would sound unnatural regardless
    if rate > 2.0:
        log.info(f"  Stretch ratio {rate:.2f} too extreme — skipping stretch")
        return None

    try:
        import librosa
        stretched = librosa.effects.time_stretch(audio.astype(np.float32), rate=rate)
        # Trim or pad to exact target length
        if len(stretched) > target_len:
            stretched = stretched[:target_len]
        elif len(stretched) < target_len:
            stretched = np.pad(stretched, (0, target_len - len(stretched)))
        return stretched.astype(np.float32)
    except ImportError:
        pass
    except Exception as e:
        log.debug(f"librosa stretch failed: {e}")

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_restoration_prompt(
        unclear_text: str,
        context_before: str,
        context_after: str,
        language: str,
        is_completely_inaudible: bool = False,
) -> str:
    lang_name = {
        "ru": "Russian", "en": "English", "de": "German",
        "fr": "French", "uk": "Ukrainian", "auto": "the detected language",
    }.get(language, language)

    if is_completely_inaudible:
        return (
            f"Audio segment is completely inaudible. Using context to suggest what was likely said.\n"
            f"Language: {lang_name}\n\n"
            f"Context before: \"{context_before}\"\n"
            f"Context after: \"{context_after}\"\n\n"
            f"Based on this context, what short phrase (1-10 words) was most likely said in the missing segment?\n"
            f"Respond ONLY with the most probable text. If impossible to determine, respond with exactly: [неразборчиво]"
        )
    else:
        return (
            f"Audio transcription contains unclear speech marked with [?] or [предположение].\n"
            f"Language: {lang_name}\n\n"
            f"Context before: \"{context_before}\"\n"
            f"Unclear text: \"{unclear_text}\"\n"
            f"Context after: \"{context_after}\"\n\n"
            f"Correct the unclear text based on context. Rules:\n"
            f"- Return ONLY corrected text, no explanation\n"
            f"- Keep close to original if reasonable\n"
            f"- If multiple options exist, use format: [вариант1/вариант2]\n"
            f"- If truly impossible, return: [неразборчиво]"
        )


def _get_context(
        segments: List[dict],
        index: int,
        window: int,
        before: bool,
) -> str:
    """Get text context around a segment."""
    texts = []

    if before:
        start = max(0, index - window)
        for i in range(start, index):
            text = segments[i].get("text", "")
            if text and INAUDIBLE_TAG not in text:
                clean = re.sub(r'\[.*?\]', '', text).strip()
                if clean:
                    texts.append(clean)
    else:
        end = min(len(segments), index + window + 1)
        for i in range(index + 1, end):
            text = segments[i].get("text", "")
            if text and INAUDIBLE_TAG not in text:
                clean = re.sub(r'\[.*?\]', '', text).strip()
                if clean:
                    texts.append(clean)

    return " ".join(texts)


# ─────────────────────────────────────────────────────────────────────────────
# Build full text
# ─────────────────────────────────────────────────────────────────────────────

def _build_full_text(segments: List[dict]) -> str:
    """Build full transcript text with timestamps header."""
    lines = []
    lines.append("TRANSCRIPT\n" + "=" * 60 + "\n")

    for seg in segments:
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        text = seg.get("text", "")
        confidence = seg.get("confidence", 0.0)
        speaker = seg.get("speaker", "")

        timestamp = f"[{_fmt_time(start)} --> {_fmt_time(end)}]"
        conf_str = f"(conf={confidence:.2f})"
        speaker_str = f" [{speaker}]" if speaker else ""

        line = f"{timestamp}{speaker_str} {conf_str}\n{text}\n"
        lines.append(line)

    full_text = "\n".join(lines)

    lines.append("\n" + "=" * 60)
    lines.append("CLEAN TRANSCRIPT (no timestamps)\n" + "=" * 60 + "\n")

    clean_parts = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if text:
            clean_parts.append(text)

    lines.append(" ".join(clean_parts))

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning utilities
# ─────────────────────────────────────────────────────────────────────────────

_NOISE_PATTERNS = [
    r"^\s*$",
    r"^\s*\.\s*$",
    r"^\s*\[.*\]\s*$",
    r"(?i)^\s*(uh+|um+|hmm+|mm+)\s*$",
    r"(?i)^\s*(music|музыка|♪|♫)\s*$",
    r"(?i)^\s*\[.*?(noise|шум|нрзб).*?\]\s*$",
    r"^\s*[.!?,;:]+\s*$",
]

_NOISE_RE = [re.compile(p) for p in _NOISE_PATTERNS]


def _is_noise_text(text: str) -> bool:
    if not text:
        return True
    for pattern in _NOISE_RE:
        if pattern.match(text):
            return True
    clean = re.sub(r'[^a-zA-Zа-яА-Я]', '', text)
    if len(clean) <= 1 and len(text) > 0:
        return True
    return False


def _clean_noise_artifacts(text: str) -> str:
    text = re.sub(r'\[(?:музыка|music|аплодисменты|applause|смех|laughter)\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'([.!?])\1+', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"