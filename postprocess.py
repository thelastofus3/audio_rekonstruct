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

    # Step 4: TTS audio restoration
    # Only for fully inaudible segments ([неразборчиво]), NOT for [предположение]
    if restore_audio and audio_path and output_audio_path:
        tts_candidates = [
            s for s in processed
            if s.get("restoration_tag") in ("inaudible", "noise")
               or (
                       s.get("restoration_method", "").startswith("llm_")
                       and s.get("restoration_tag") not in ("assumption", "low_confidence")
               )
        ]
        if tts_candidates:
            log.info(
                f"\n[TTS AUDIO RESTORATION] Synthesizing and splicing "
                f"{len(tts_candidates)} fully-inaudible segment(s) into audio..."
            )
            _restore_audio_tts(processed, audio_path, output_audio_path, tts_language, log)
        else:
            log.info("No fully-inaudible segments found — audio unchanged.")
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


# =============================================================================
# Marking unclear segments
# =============================================================================

def _mark_unclear_segments(
        segments: List[dict],
        threshold: float,
        log: logging.Logger,
) -> List[dict]:
    result = []

    for seg in segments:
        seg = dict(seg)
        text = seg.get("text", "").strip()
        confidence = seg.get("confidence", 1.0)
        no_speech_prob = seg.get("no_speech_prob", 0.0)

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
            seg["is_unclear"] = True
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


# =============================================================================
# LLM restoration
# =============================================================================

def _llm_restore(
        segments: List[dict],
        provider: str,
        language: str,
        log: logging.Logger,
) -> List[dict]:
    llm_fn = _get_llm_function(provider, log)
    if not llm_fn:
        log.warning("LLM not available, skipping restoration.")
        return segments

    result = list(segments)
    _unclear_word_re = re.compile(r'\[[^\]]+\?\]')

    for i, seg in enumerate(result):
        seg_text = seg.get("text", "")

        has_unclear_words = bool(_unclear_word_re.search(seg_text))
        if not seg.get("is_unclear") and not has_unclear_words:
            continue

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
                    slot_duration = seg.get("end", 0.0) - seg.get("start", 0.0)
                    word_count = len(restored.split())
                    # Reject if too few words for a long slot (< 1 word per 1.5s)
                    if slot_duration > 2.0 and word_count < max(2, slot_duration / 1.5):
                        log.info(
                            f"LLM restored only {word_count} word(s) for "
                            f"{slot_duration:.1f}s slot — keeping original unclear tag."
                        )
                    else:
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
    import requests as _requests

    base_url = os.environ.get("LOCAL_LLM_URL", "http://localhost:11434/v1")
    model = os.environ.get("LOCAL_LLM_MODEL", "")

    if not model:
        model = _detect_ollama_model(base_url, log)
        if not model:
            log.warning(
                "No local LLM model found. "
                "Install Ollama (https://ollama.com) and run: ollama pull qwen2.5:7b"
            )
            return None

    log.info(f"Local LLM: {base_url}, model={model}")

    ollama_base = base_url.replace("/v1", "")
    try:
        ping = _requests.get(f"{ollama_base}/api/version", timeout=5)
        if ping.status_code != 200:
            raise ConnectionError(f"Ollama ping returned {ping.status_code}")
        log.info(f"Ollama is running (version: {ping.json().get('version', '?')})")
    except Exception as e:
        log.warning(f"Cannot reach Ollama at {ollama_base}: {e}")
        return None

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
    import requests as _requests

    ollama_base = base_url.replace("/v1", "")
    preferred = ["qwen2.5", "gemma3", "mistral", "llama3", "phi3", "llama2"]

    try:
        resp = _requests.get(f"{ollama_base}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            if models:
                log.info(f"Ollama installed models: {', '.join(models)}")
                for pref in preferred:
                    for m in models:
                        if m.startswith(pref):
                            return m
                return models[0]
    except Exception:
        pass

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


# =============================================================================
# TTS audio restoration
# =============================================================================

def _restore_audio_tts(
        segments: List[dict],
        source_audio_path: str,
        output_audio_path: str,
        tts_language: str,
        log: logging.Logger,
):
    """
    Splice TTS audio only for fully inaudible segments.
    Segments tagged assumption/low_confidence are SKIPPED — original audio is kept.
    """
    import numpy as np
    import scipy.io.wavfile as wav_io
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
        for seg_index, seg in enumerate(segments):
            seg_text_raw = seg.get("text", "")
            restoration_tag = seg.get("restoration_tag", "")

            # SKIP partial-speech segments — do not overwrite audible speech
            if restoration_tag in ("assumption", "low_confidence"):
                log.info(
                    f"  Segment {seg.get('id','?')} ({seg.get('start',0):.1f}s): "
                    f"partial speech present (tag={restoration_tag}) — keeping original audio."
                )
                continue

            was_llm_restored = seg.get("restoration_method", "").startswith("llm_")
            is_inaudible = restoration_tag in ("inaudible", "noise")

            if not was_llm_restored and not is_inaudible:
                continue

            # Build clean text for TTS (strip all tags/brackets)
            text = seg_text_raw
            text = text.replace(ASSUMPTION_TAG, "").replace(INAUDIBLE_TAG, "").strip()
            text = _variant_re.sub(r'\1', text)       # [A/B] → A
            text = _unclear_word_re2.sub(r'\1', text)  # [word?] → word
            text = re.sub(r'[\[\]]', '', text).strip() # remove remaining brackets

            if not text:
                log.info(
                    f"  Segment {seg.get('id','?')} ({seg.get('start',0):.1f}s): "
                    f"fully inaudible, silence kept."
                )
                continue

            prev_end = float(segments[seg_index - 1].get("end", 0.0)) if seg_index > 0 else 0.0
            next_start = (
                float(segments[seg_index + 1].get("start", len(audio) / sr))
                if seg_index + 1 < len(segments)
                else len(audio) / sr
            )
            seg_start, seg_end = _estimate_tts_splice_bounds(
                seg,
                audio,
                sr,
                log,
                prev_end=prev_end,
                next_start=next_start,
            )
            seg_duration = seg_end - seg_start
            if seg_duration <= 0.05:
                continue

            # Synthesize TTS
            tts_wav = os.path.join(tmpdir, f"tts_{seg.get('id', restored_count)}.wav")
            try:
                tts_fn(text, tts_wav)
            except Exception as e:
                log.warning(f"TTS synthesis failed for segment {seg.get('id','?')}: {e}")
                continue

            if not os.path.exists(tts_wav) or os.path.getsize(tts_wav) == 0:
                continue

            tts_sr, tts_data = wav_io.read(tts_wav)
            if tts_data.dtype == np.int16:
                tts_audio = tts_data.astype(np.float32) / 32768.0
            else:
                tts_audio = tts_data.astype(np.float32)

            if tts_audio.ndim > 1:
                tts_audio = tts_audio.mean(axis=1)

            if tts_sr != sr:
                tts_audio = _resample(tts_audio, tts_sr, sr)

            tts_len = len(tts_audio)
            target_samples = int(seg_duration * sr)
            ratio = tts_len / max(target_samples, 1)

            log.info(
                f"  Segment {seg.get('id','?')} slot={seg_duration:.2f}s, "
                f"TTS={tts_len/sr:.2f}s, ratio={ratio:.2f}"
            )

            # Fit TTS into the slot
            if ratio <= 1.25:
                if tts_len < target_samples:
                    # Pad with silence for inaudible segments to avoid bleed-through
                    pad_needed = target_samples - tts_len
                    silence = np.zeros(pad_needed, dtype=tts_audio.dtype)
                    tts_audio = np.concatenate([tts_audio, silence])
                else:
                    tts_audio = tts_audio[:target_samples]
            else:
                stretched = _pitch_preserving_stretch(tts_audio, sr, target_samples, log)
                if stretched is not None:
                    tts_audio = stretched
                    log.info(f"  Used pitch-preserving stretch (ratio={ratio:.2f})")
                else:
                    log.info(
                        f"  TTS longer than slot by {(tts_len-target_samples)/sr:.2f}s "
                        f"— inserting at natural length"
                    )
                    splice_start = int(seg_start * sr)
                    gap_end = int(seg_end * sr)
                    audio = _splice_with_crossfade(audio, tts_audio, splice_start, gap_end, sr)
                    restored_count += 1
                    log.info(
                        f"  Spliced TTS for segment {seg.get('id','?')} "
                        f"({seg_start:.1f}s): \"{text[:50]}\""
                    )
                    continue

            # Match volume to surrounding audio
            start_sample = int(seg_start * sr)
            end_sample = min(int(seg_end * sr), len(audio))
            if end_sample > start_sample and end_sample <= len(audio):
                source_rms = np.sqrt(np.mean(audio[start_sample:end_sample] ** 2))
                tts_rms = np.sqrt(np.mean(tts_audio ** 2))
                if tts_rms > 1e-6 and source_rms > 1e-6:
                    tts_audio = tts_audio * (source_rms / tts_rms)

            # Splice
            splice_start = int(seg_start * sr)
            splice_end = min(int(seg_end * sr), len(audio))
            audio = _splice_with_crossfade(audio, tts_audio, splice_start, splice_end, sr)
            restored_count += 1
            log.info(
                f"  Spliced TTS for segment {seg.get('id','?')} "
                f"({seg_start:.1f}s-{seg_end:.1f}s): \"{text[:40]}\""
            )

    if restored_count == 0:
        log.warning("No segments spliced — copying source audio unchanged.")
        import shutil
        shutil.copy(source_audio_path, output_audio_path)
        return

    audio_out = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_out * 32767).astype(np.int16)
    wav_io.write(output_audio_path, sr, audio_int16)
    log.info(f"Restored audio saved: {output_audio_path} ({restored_count} segments spliced)")


def _get_tts_function(language: str, log: logging.Logger):
    """Return a TTS function: tts_fn(text, output_wav_path)."""

    # edge-tts: Microsoft Neural TTS — best quality, free, needs internet
    try:
        import edge_tts
        import asyncio
        import subprocess

        EDGE_VOICES = {
            "ru": "ru-RU-SvetlanaNeural",
            "en": "en-US-AndrewNeural",
            "de": "de-DE-KatjaNeural",
            "fr": "fr-FR-DeniseNeural",
            "uk": "uk-UA-PolinaNeural",
        }
        voice = EDGE_VOICES.get(language, "en-US-AndrewNeural")

        def edge_synthesize(text: str, output_wav: str):
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_mp3 = f.name
            try:
                async def _run():
                    communicate = edge_tts.Communicate(text, voice)
                    await communicate.save(tmp_mp3)

                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(_run())
                finally:
                    loop.close()

                # Convert MP3 -> WAV and trim leading silence
                result = subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", tmp_mp3,
                        "-af", "silenceremove=start_periods=1:start_silence=0.02:start_threshold=-50dB",
                        "-ar", "16000", "-ac", "1",
                        "-acodec", "pcm_s16le", output_wav,
                    ],
                    capture_output=True, timeout=30,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"ffmpeg failed: {result.stderr[-300:]}")
            finally:
                if os.path.exists(tmp_mp3):
                    os.unlink(tmp_mp3)

        log.info(f"TTS backend: edge-tts (Microsoft Neural, voice={voice})")
        return edge_synthesize
    except ImportError:
        log.info("edge-tts not available, trying gTTS...")

    # gTTS fallback
    try:
        from gtts import gTTS
        import subprocess

        def gtts_synthesize(text: str, output_wav: str):
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_mp3 = f.name
            try:
                tts = gTTS(text=text, lang=language, slow=False)
                tts.save(tmp_mp3)
                # Trim leading silence that causes the ~1s delay
                result = subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", tmp_mp3,
                        "-af", "silenceremove=start_periods=1:start_silence=0.02:start_threshold=-50dB",
                        "-ar", "16000", "-ac", "1",
                        "-acodec", "pcm_s16le", output_wav,
                    ],
                    capture_output=True, timeout=30,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"ffmpeg MP3->WAV failed: {result.stderr[-300:]}")
            finally:
                if os.path.exists(tmp_mp3):
                    os.unlink(tmp_mp3)

        log.info("TTS backend: gTTS (Google TTS)")
        return gtts_synthesize
    except ImportError:
        log.info("gTTS not available, trying pyttsx3...")

    # pyttsx3 offline fallback
    try:
        import pyttsx3

        engine = pyttsx3.init()
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
        log.warning("No TTS backend found. Install: pip install edge-tts")
        return None


def _resample(audio: "np.ndarray", from_sr: int, to_sr: int) -> "np.ndarray":
    import numpy as np
    if from_sr == to_sr:
        return audio
    try:
        import scipy.signal
        n_samples = int(len(audio) * to_sr / from_sr)
        return scipy.signal.resample(audio, n_samples).astype(np.float32)
    except Exception:
        return audio


def _estimate_tts_splice_bounds(
        seg: dict,
        audio: "np.ndarray",
        sr: int,
        log: "logging.Logger",
        prev_end: Optional[float] = None,
        next_start: Optional[float] = None,
) -> Tuple[float, float]:
    seg_start = float(seg.get("start", 0.0) or 0.0)
    seg_end = float(seg.get("end", 0.0) or 0.0)
    words = seg.get("words", []) or []
    was_llm_restored = str(seg.get("restoration_method", "")).startswith("llm_")
    prev_end = seg_start if prev_end is None else float(prev_end)
    next_start = seg_end if next_start is None else float(next_start)

    if seg_end <= seg_start:
        return seg_start, seg_end

    if was_llm_restored:
        onset_search_start = max(0.0, prev_end)
        onset_search_end = min(next_start, seg_start + 0.12)
        offset_search_start = max(onset_search_start, min(seg_end - 0.08, prev_end))
        offset_search_end = min(len(audio) / sr, max(seg_end, next_start))

        onset = _find_speech_transition(
            audio=audio,
            sr=sr,
            search_start_sec=onset_search_start,
            search_end_sec=onset_search_end,
            transition="onset",
        )
        offset = _find_speech_transition(
            audio=audio,
            sr=sr,
            search_start_sec=offset_search_start,
            search_end_sec=offset_search_end,
            transition="offset",
        )
        start_candidate = min(onset, seg_start)
        # Blend between the earlier detected onset and the ASR start. This keeps
        # the cut earlier than Whisper's late timestamp without jumping all the
        # way to the first transient in the gap.
        refined_start = max(prev_end, start_candidate + (seg_start - start_candidate) * 0.40)
        refined_end = max(seg_end, offset + 0.02)

        log.info(
            f"  Segment {seg.get('id','?')}: full-segment replacement "
            f"{seg_start:.3f}-{seg_end:.3f}s -> {refined_start:.3f}-{refined_end:.3f}s"
        )
        return refined_start, refined_end

    candidate_start = seg_start
    candidate_end = seg_end

    unclear_words = [w for w in words if w.get("is_unclear", False)]
    if unclear_words:
        candidate_start = float(unclear_words[0].get("start", candidate_start) or candidate_start)
        candidate_end = float(unclear_words[-1].get("end", candidate_end) or candidate_end)
        if len(unclear_words) == len(words):
            candidate_start = seg_start
            candidate_end = seg_end
    elif words:
        candidate_start = float(words[0].get("start", candidate_start) or candidate_start)
        candidate_end = float(words[-1].get("end", candidate_end) or candidate_end)

    candidate_start = max(seg_start, candidate_start - 0.06)
    candidate_end = min(seg_end, candidate_end + 0.04)

    refined_start = _find_energy_boundary(
        audio=audio,
        sr=sr,
        anchor_sec=candidate_start,
        search_start_sec=seg_start,
        search_end_sec=min(seg_end, candidate_start + 0.35),
        direction="backward",
    )
    refined_end = _find_energy_boundary(
        audio=audio,
        sr=sr,
        anchor_sec=candidate_end,
        search_start_sec=max(seg_start, candidate_end - 0.35),
        search_end_sec=seg_end,
        direction="forward",
    )

    if refined_end - refined_start < 0.08:
        refined_start = max(seg_start, min(candidate_start, seg_end - 0.08))
        refined_end = min(seg_end, max(candidate_end, refined_start + 0.08))

    if abs(refined_start - seg_start) > 0.02 or abs(refined_end - seg_end) > 0.02:
        log.info(
            f"  Segment {seg.get('id','?')}: splice bounds "
            f"{seg_start:.3f}-{seg_end:.3f}s -> {refined_start:.3f}-{refined_end:.3f}s"
        )

    return refined_start, refined_end


def _find_speech_transition(
        audio: "np.ndarray",
        sr: int,
        search_start_sec: float,
        search_end_sec: float,
        transition: str,
) -> float:
    import numpy as np

    search_start_sec = max(0.0, search_start_sec)
    search_end_sec = min(len(audio) / sr, search_end_sec)
    if search_end_sec <= search_start_sec:
        return search_start_sec

    start_sample = int(search_start_sec * sr)
    end_sample = int(search_end_sec * sr)
    window = max(128, int(0.02 * sr))
    hop = max(64, int(0.005 * sr))
    if end_sample - start_sample < window:
        return search_start_sec if transition == "onset" else search_end_sec

    rms = []
    centers = []
    for pos in range(start_sample, end_sample - window + 1, hop):
        frame = audio[pos:pos + window]
        rms.append(float(np.sqrt(np.mean(frame ** 2) + 1e-12)))
        centers.append((pos + window / 2) / sr)

    if not rms:
        return search_start_sec if transition == "onset" else search_end_sec

    rms = np.asarray(rms, dtype=np.float32)
    noise_floor = float(np.percentile(rms, 15))
    speech_peak = float(np.percentile(rms, 95))
    threshold = noise_floor + (speech_peak - noise_floor) * 0.18
    min_run = 3

    if transition == "onset":
        run = 0
        for idx, level in enumerate(rms):
            run = run + 1 if level >= threshold else 0
            if run >= min_run:
                return centers[max(0, idx - min_run + 1)]
        return search_start_sec

    run = 0
    for idx in range(len(rms) - 1, -1, -1):
        level = rms[idx]
        run = run + 1 if level >= threshold else 0
        if run >= min_run:
            return centers[min(len(centers) - 1, idx + min_run - 1)]
    return search_end_sec


def _find_energy_boundary(
        audio: "np.ndarray",
        sr: int,
        anchor_sec: float,
        search_start_sec: float,
        search_end_sec: float,
        direction: str,
) -> float:
    import numpy as np

    if search_end_sec <= search_start_sec:
        return max(search_start_sec, min(anchor_sec, search_end_sec))

    start_sample = max(0, int(search_start_sec * sr))
    end_sample = min(len(audio), int(search_end_sec * sr))
    window = max(64, int(0.012 * sr))
    hop = max(32, int(0.004 * sr))

    if end_sample - start_sample < window:
        return max(search_start_sec, min(anchor_sec, search_end_sec))

    rms = []
    centers = []
    for pos in range(start_sample, end_sample - window + 1, hop):
        frame = audio[pos:pos + window]
        rms.append(float(np.sqrt(np.mean(frame ** 2) + 1e-12)))
        centers.append((pos + window / 2) / sr)

    if not rms:
        return max(search_start_sec, min(anchor_sec, search_end_sec))

    rms = np.asarray(rms, dtype=np.float32)
    floor = float(np.percentile(rms, 20))
    peak = float(np.percentile(rms, 90))
    threshold = floor + (peak - floor) * 0.22

    if direction == "backward":
        boundary = anchor_sec
        for center_sec, level in zip(reversed(centers), reversed(rms)):
            if center_sec > anchor_sec:
                continue
            if level <= threshold:
                boundary = center_sec
                break
        return max(search_start_sec, min(boundary, search_end_sec))

    boundary = anchor_sec
    for center_sec, level in zip(centers, rms):
        if center_sec < anchor_sec:
            continue
        if level <= threshold:
            boundary = center_sec
            break
    return max(search_start_sec, min(boundary, search_end_sec))


def _splice_with_crossfade(
        audio: "np.ndarray",
        insert_audio: "np.ndarray",
        splice_start: int,
        splice_end: int,
        sr: int,
) -> "np.ndarray":
    import numpy as np

    splice_start = max(0, splice_start)
    splice_end = max(splice_start, splice_end)
    replaced_len = splice_end - splice_start
    insert_len = len(insert_audio)

    crossfade = min(
        int(0.025 * sr),
        max(0, replaced_len // 3),
        max(0, insert_len // 3),
    )

    if crossfade <= 8:
        output_end = splice_start + insert_len
        if output_end > len(audio):
            audio = np.pad(audio, (0, output_end - len(audio)))
        audio[splice_start:output_end] = insert_audio
        return audio

    left_ctx_start = max(0, splice_start - crossfade)
    left_ctx = audio[left_ctx_start:splice_start].copy()
    right_ctx_end = min(len(audio), splice_end + crossfade)
    right_ctx = audio[splice_end:right_ctx_end].copy()

    if len(left_ctx) < crossfade:
        left_ctx = np.pad(left_ctx, (crossfade - len(left_ctx), 0))
    if len(right_ctx) < crossfade:
        right_ctx = np.pad(right_ctx, (0, crossfade - len(right_ctx)))

    fade_out = np.sqrt(np.linspace(1.0, 0.0, crossfade, dtype=np.float32))
    fade_in = np.sqrt(np.linspace(0.0, 1.0, crossfade, dtype=np.float32))

    head = left_ctx[:crossfade] * fade_out + insert_audio[:crossfade] * fade_in
    tail = insert_audio[-crossfade:] * fade_out + right_ctx[:crossfade] * fade_in
    middle = insert_audio[crossfade:-crossfade]

    return np.concatenate([
        audio[:left_ctx_start],
        head,
        middle,
        tail,
        audio[right_ctx_end:],
    ])


def _pitch_preserving_stretch(
        audio: "np.ndarray",
        sr: int,
        target_len: int,
        log: "logging.Logger",
) -> "Optional[np.ndarray]":
    import numpy as np

    if len(audio) == target_len:
        return audio

    rate = len(audio) / max(target_len, 1)

    if rate > 2.0:
        log.info(f"  Stretch ratio {rate:.2f} too extreme — skipping stretch")
        return None

    try:
        import librosa
        stretched = librosa.effects.time_stretch(audio.astype(np.float32), rate=rate)
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


# =============================================================================
# Prompt builder
# =============================================================================

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


# =============================================================================
# Build full text
# =============================================================================

def _build_full_text(segments: List[dict]) -> str:
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


# =============================================================================
# Text cleaning utilities
# =============================================================================

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
