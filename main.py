"""
# file: main.py
Speech restoration pipeline from low-quality video.
Usage: python main.py --input video.mp4 --output out_folder --language ru
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from extract_audio import extract_audio
from enhance_audio import enhance_audio
from vad import run_vad
from asr import run_asr
from postprocess import postprocess_transcript


def setup_logging(output_dir: Path) -> logging.Logger:
    log_file = output_dir / f"process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("speech_restore")
    logger.info(f"Log file: {log_file}")
    return logger


def check_dependencies() -> bool:
    """Check that required system tools are available."""
    import shutil

    missing = []

    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg (https://ffmpeg.org/download.html)")

    if missing:
        print("=" * 60)
        print("MISSING DEPENDENCIES:")
        for m in missing:
            print(f"  - {m}")
        print("=" * 60)
        return False
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Speech restoration from low-quality video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input video.mp4 --output ./results --language ru
  python main.py --input lecture.mkv --output ./out --language en --diarize
  python main.py --input interview.avi --output ./out --language auto --no-enhance
        """,
    )
    parser.add_argument("--input", required=True, help="Input video file (mp4/mkv/avi)")
    parser.add_argument("--output", required=True, help="Output folder for results")
    parser.add_argument(
        "--language",
        default="auto",
        help="Language code (ru, en, de, ...) or 'auto' for auto-detection",
    )
    parser.add_argument(
        "--model",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size (default: medium)",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Enable speaker diarization (requires pyannote.audio)",
    )
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Skip audio enhancement step",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Skip VAD segmentation (process full audio)",
    )
    parser.add_argument(
        "--llm-postprocess",
        action="store_true",
        help="Use LLM to restore unclear segments (requires API key)",
    )
    parser.add_argument(
        "--llm-provider",
        default="local",
        choices=["local", "claude", "openai", "anthropic"],
        help="LLM provider: local (Ollama, free), claude, openai (default: local)",
    )
    parser.add_argument(
        "--restore-audio",
        action="store_true",
        help="Synthesize restored text via TTS and splice back into audio (requires edge-tts or gtts)",
    )
    parser.add_argument(
        "--tts-language",
        default=None,
        help="Language code for TTS synthesis when --restore-audio is used (default: input language or en)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Confidence threshold below which segment is marked unclear (0.0-1.0)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for inference (default: auto)",
    )
    return parser.parse_args()


def save_results(
        output_dir: Path,
        transcript_segments: list,
        full_text: str,
        logger: logging.Logger,
):
    """Save transcript.txt and transcript.json."""

    txt_path = output_dir / "transcript.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    logger.info(f"Transcript saved: {txt_path}")

    json_path = output_dir / "transcript.json"
    output_data = {
        "created_at": datetime.now().isoformat(),
        "total_segments": len(transcript_segments),
        "segments": transcript_segments,
        "full_text": full_text,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON transcript saved: {json_path}")

    return txt_path, json_path


def mux_video_with_audio(
        video_path: Path,
        audio_path: Path,
        output_video_path: Path,
        logger: logging.Logger,
) -> Path:
    """Combine original video with processed audio."""
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(output_video_path),
    ]
    logger.info(f"Muxing processed audio back into video: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        logger.error(f"ffmpeg stderr:\n{result.stderr[-2000:]}")
        raise RuntimeError(
            f"ffmpeg video mux failed with code {result.returncode}.\n"
            f"stderr: {result.stderr[-500:]}"
        )
    if not output_video_path.exists() or output_video_path.stat().st_size == 0:
        raise RuntimeError(f"Output video is empty or missing: {output_video_path}")
    logger.info(f"Output video saved: {output_video_path}")
    return output_video_path


def main():
    print("\n" + "=" * 60)
    print("  SPEECH RESTORATION PIPELINE")
    print("=" * 60 + "\n")

    if not check_dependencies():
        sys.exit(1)

    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    supported_ext = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
    if input_path.suffix.lower() not in supported_ext:
        print(f"[WARNING] Unsupported extension '{input_path.suffix}'. Proceeding anyway.")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info(f"Input: {input_path.resolve()}")
    logger.info(f"Output: {output_dir.resolve()}")
    logger.info(f"Language: {args.language}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {args.device}")

    total_start = time.time()

    # STEP 1: Extract audio
    logger.info("\n[STEP 1/5] Extracting audio from video...")
    raw_audio_path = output_dir / "audio_raw.wav"
    try:
        extract_audio(str(input_path), str(raw_audio_path), logger=logger)
        logger.info(f"Raw audio saved: {raw_audio_path}")
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")
        sys.exit(1)

    # STEP 2: Enhance audio
    if not args.no_enhance:
        logger.info("\n[STEP 2/5] Enhancing audio (denoising, dereverberation)...")
        enhanced_audio_path = output_dir / "audio_enhanced.wav"
        try:
            enhance_audio(
                str(raw_audio_path),
                str(enhanced_audio_path),
                logger=logger,
            )
            logger.info(f"Enhanced audio saved: {enhanced_audio_path}")
            asr_input = str(enhanced_audio_path)
        except Exception as e:
            logger.warning(f"Enhancement failed: {e}. Using raw audio.")
            import shutil
            shutil.copy(str(raw_audio_path), str(output_dir / "audio_enhanced.wav"))
            asr_input = str(raw_audio_path)
    else:
        logger.info("\n[STEP 2/5] Skipping audio enhancement (--no-enhance).")
        asr_input = str(raw_audio_path)
        import shutil
        shutil.copy(str(raw_audio_path), str(output_dir / "audio_enhanced.wav"))

    # STEP 3: VAD
    if not args.no_vad:
        logger.info("\n[STEP 3/5] Running Voice Activity Detection...")
        try:
            vad_segments = run_vad(asr_input, logger=logger)
            logger.info(f"VAD found {len(vad_segments)} speech segments")
        except Exception as e:
            logger.warning(f"VAD failed: {e}. Will process full audio.")
            vad_segments = None
    else:
        logger.info("\n[STEP 3/5] Skipping VAD (--no-vad).")
        vad_segments = None

    # STEP 4: ASR
    logger.info("\n[STEP 4/5] Running ASR (Speech Recognition)...")
    try:
        language = None if args.language == "auto" else args.language
        asr_segments = run_asr(
            audio_path=asr_input,
            model_name=args.model,
            language=language,
            device=args.device,
            vad_segments=vad_segments,
            confidence_threshold=args.confidence_threshold,
            logger=logger,
        )
        logger.info(f"ASR produced {len(asr_segments)} segments")
    except Exception as e:
        logger.error(f"ASR failed: {e}")
        sys.exit(1)

    # STEP 5: Postprocessing
    logger.info("\n[STEP 5/5] Postprocessing transcript...")
    try:
        restored_audio_path = str(output_dir / "audio_restored.wav")
        do_restore_audio = args.llm_postprocess or args.restore_audio
        tts_lang = args.tts_language or (args.language if args.language != "auto" else "en")
        final_segments, full_text = postprocess_transcript(
            segments=asr_segments,
            use_llm=args.llm_postprocess,
            llm_provider=args.llm_provider,
            confidence_threshold=args.confidence_threshold,
            language=args.language,
            logger=logger,
            restore_audio=do_restore_audio,
            audio_path=asr_input,
            output_audio_path=restored_audio_path,
            tts_language=tts_lang,
        )
    except Exception as e:
        logger.warning(f"Postprocessing error: {e}. Using raw ASR output.")
        final_segments = asr_segments
        full_text = " ".join(s.get("text", "") for s in asr_segments)

    # Save results
    final_audio_path = output_dir / "audio_enhanced.wav"
    if args.llm_postprocess or args.restore_audio:
        restored_candidate = output_dir / "audio_restored.wav"
        if restored_candidate.exists() and restored_candidate.stat().st_size > 0:
            final_audio_path = restored_candidate

    output_video_path = output_dir / f"{input_path.stem}_fixed{input_path.suffix}"
    try:
        mux_video_with_audio(input_path, final_audio_path, output_video_path, logger)
    except Exception as e:
        logger.warning(f"Video mux failed: {e}")
        output_video_path = None

    txt_path, json_path = save_results(output_dir, final_segments, full_text, logger)

    total_elapsed = time.time() - total_start

    print("\n" + "=" * 60)
    print("  DONE!")
    print("=" * 60)
    print(f"  Time elapsed   : {total_elapsed:.1f}s")
    print(f"  Segments found : {len(final_segments)}")
    print(f"  Output folder  : {output_dir.resolve()}")
    if output_video_path:
        print(f"  Output video   : {output_video_path}")
    print(f"  Enhanced audio : {output_dir / 'audio_enhanced.wav'}")
    if args.llm_postprocess or args.restore_audio:
        print(f"  Restored audio : {output_dir / 'audio_restored.wav'}")
    print(f"  Transcript TXT : {txt_path}")
    print(f"  Transcript JSON: {json_path}")
    print("=" * 60 + "\n")

    logger.info(f"Pipeline complete in {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
