"""
# file: postprocess.py
Transcript postprocessing:
  - Mark unclear/inaudible segments with [неразборчиво], [предположение], [вариант1/вариант2]
  - Optional LLM restoration of unclear text
  - Build clean full-text transcript
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
    llm_provider: str = "openai",
    confidence_threshold: float = 0.5,
    language: str = "auto",
    logger: logging.Logger = None,
) -> Tuple[List[dict], str]:
    """
    Postprocess ASR segments: mark unclear parts, optionally use LLM.

    Args:
        segments: List of ASR segment dicts.
        use_llm: Whether to use LLM for unclear restoration.
        llm_provider: 'openai', 'anthropic', or 'local'.
        confidence_threshold: Threshold below which segment is unclear.
        language: Language code for LLM prompting.
        logger: Optional logger.

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
        unclear_count = sum(1 for s in processed if s.get("is_unclear"))
        if unclear_count > 0:
            log.info(f"Running LLM restoration on {unclear_count} unclear segments...")
            processed = _llm_restore(processed, llm_provider, language, log)
        else:
            log.info("No unclear segments for LLM restoration.")

    # Step 4: Build full text
    full_text = _build_full_text(processed)

    # Step 5: Statistics
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
                # Wrap as assumption
                clean_text = _clean_noise_artifacts(text)
                seg["text"] = f"{ASSUMPTION_TAG} {clean_text}"
                seg["restoration_tag"] = "assumption"

        # Check for noise-only text
        elif _is_noise_text(text):
            seg["text"] = INAUDIBLE_TAG
            seg["is_unclear"] = True
            seg["restoration_tag"] = "noise"

        # Low confidence but has text
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

        # Check if segment already has unclear markers
        if INAUDIBLE_TAG in seg.get("text", "") or ASSUMPTION_TAG in seg.get("text", ""):
            result.append(seg)
            continue

        # Rebuild text with word-level confidence markers
        new_words = []
        has_unclear_words = False

        for w in words:
            word = w.get("word", "")
            word_conf = w.get("confidence", 1.0)

            if word_conf < confidence_threshold and word.strip():
                # Very low confidence — mark as unclear
                marked = w.copy()
                marked["is_unclear"] = True
                new_words.append(marked)
                has_unclear_words = True
            else:
                new_words.append(w)

        if has_unclear_words:
            # Rebuild text with variant markers for unclear words
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

    # Get LLM function
    llm_fn = _get_llm_function(provider, log)
    if not llm_fn:
        log.warning("LLM not available, skipping restoration.")
        return segments

    result = list(segments)

    # Process unclear segments with context window
    for i, seg in enumerate(result):
        if not seg.get("is_unclear"):
            continue

        if seg.get("text") == INAUDIBLE_TAG:
            # Try to restore completely inaudible segment
            context_before = _get_context(result, i, window=3, before=True)
            context_after = _get_context(result, i, window=2, before=False)

            if not context_before and not context_after:
                continue  # No context available

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
            # Try to clarify assumption
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
    """Get LLM callable based on provider."""

    if provider == "openai":
        try:
            import openai
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                log.warning("OPENAI_API_KEY not set.")
                return None

            client = openai.OpenAI(api_key=api_key)

            def call_openai(prompt: str) -> str:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a professional transcription editor. Restore unclear speech based on context. Return ONLY the restored text, no explanations."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=200,
                    temperature=0.3,
                )
                return response.choices[0].message.content.strip()

            return call_openai
        except ImportError:
            log.warning("openai package not installed.")
            return None

    elif provider == "anthropic":
        try:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                log.warning("ANTHROPIC_API_KEY not set.")
                return None

            client = anthropic.Anthropic(api_key=api_key)

            def call_anthropic(prompt: str) -> str:
                message = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=200,
                    system="You are a professional transcription editor. Restore unclear speech based on context. Return ONLY the restored text, no explanations.",
                    messages=[{"role": "user", "content": prompt}],
                )
                return message.content[0].text.strip()

            return call_anthropic
        except ImportError:
            log.warning("anthropic package not installed.")
            return None

    elif provider == "local":
        # Try Ollama or local OpenAI-compatible endpoint
        try:
            import requests
            base_url = os.environ.get("LOCAL_LLM_URL", "http://localhost:11434/v1")

            def call_local(prompt: str) -> str:
                resp = requests.post(
                    f"{base_url}/chat/completions",
                    json={
                        "model": os.environ.get("LOCAL_LLM_MODEL", "llama3"),
                        "messages": [
                            {"role": "system", "content": "Restore unclear speech from context. Return only the text."},
                            {"role": "user", "content": prompt},
                        ],
                        "max_tokens": 200,
                        "temperature": 0.3,
                    },
                    timeout=30,
                )
                return resp.json()["choices"][0]["message"]["content"].strip()

            return call_local
        except Exception as e:
            log.warning(f"Local LLM not available: {e}")
            return None

    return None


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
        return f"""Audio segment is completely inaudible. Using context to suggest what was likely said.
Language: {lang_name}

Context before: "{context_before}"
Context after: "{context_after}"

Based on this context, what short phrase (1-10 words) was most likely said in the missing segment?
Respond ONLY with the most probable text. If impossible to determine, respond with exactly: [неразборчиво]"""
    else:
        return f"""Audio transcription contains unclear speech marked with [?] or [предположение].
Language: {lang_name}

Context before: "{context_before}"
Unclear text: "{unclear_text}"
Context after: "{context_after}"

Correct the unclear text based on context. Rules:
- Return ONLY corrected text, no explanation
- Keep close to original if reasonable
- If multiple options exist, use format: [вариант1/вариант2]
- If truly impossible, return: [неразборчиво]"""


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

    # Also add clean transcript at the end
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

# Common noise/artifact patterns from Whisper
_NOISE_PATTERNS = [
    r"^\s*$",                           # empty
    r"^\s*\.\s*$",                      # just a dot
    r"^\s*\[.*\]\s*$",                  # just brackets
    r"(?i)^\s*(uh+|um+|hmm+|mm+)\s*$", # filler sounds
    r"(?i)^\s*(music|музыка|♪|♫)\s*$",  # music annotations
    r"(?i)^\s*\[.*?(noise|шум|нрзб).*?\]\s*$",  # noise markers
    r"^\s*[.!?,;:]+\s*$",              # just punctuation
]

_NOISE_RE = [re.compile(p) for p in _NOISE_PATTERNS]


def _is_noise_text(text: str) -> bool:
    """Check if text is just noise/artifact."""
    if not text:
        return True
    for pattern in _NOISE_RE:
        if pattern.match(text):
            return True
    # Very short suspicious text
    clean = re.sub(r'[^a-zA-Zа-яА-Я]', '', text)
    if len(clean) <= 1 and len(text) > 0:
        return True
    return False


def _clean_noise_artifacts(text: str) -> str:
    """Remove common Whisper hallucination patterns."""
    # Remove annotation brackets that whisper sometimes adds
    text = re.sub(r'\[(?:музыка|music|аплодисменты|applause|смех|laughter)\]', '', text, flags=re.IGNORECASE)
    # Remove duplicate punctuation
    text = re.sub(r'([.!?])\1+', r'\1', text)
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _fmt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"
