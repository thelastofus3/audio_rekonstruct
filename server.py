"""
server.py — Web UI for Speech Restoration Pipeline
Run:  python server.py
Open: http://localhost:5000
"""

import json
import os
import queue
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file

app = Flask(__name__)

UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# job_id → {"status", "log_queue", "output_dir"}
JOBS: dict = {}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["video"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    job_id  = str(uuid.uuid4())[:8]
    job_dir = RESULTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    upload_path = UPLOAD_DIR / f"{job_id}_{f.filename}"
    f.save(str(upload_path))

    # Read form options
    language     = request.form.get("language",     "auto")
    model        = request.form.get("model",        "medium")
    llm_provider = request.form.get("llm_provider", "local")
    llm_pp       = request.form.get("llm_postprocess", "0") == "1"
    restore      = request.form.get("restore_audio",   "0") == "1"
    no_enhance   = request.form.get("no_enhance",      "0") == "1"

    # Build main.py command
    cmd = [
        sys.executable, "main.py",
        "--input",       str(upload_path),
        "--output",      str(job_dir),
        "--language",    language,
        "--model",       model,
        "--llm-provider", llm_provider,
    ]
    if llm_pp:     cmd.append("--llm-postprocess")
    if restore:    cmd.append("--restore-audio")
    if no_enhance: cmd.append("--no-enhance")

    # Register job
    q = queue.Queue()
    JOBS[job_id] = {
        "status":    "running",
        "log_queue": q,
        "output_dir": str(job_dir),
    }

    # Run pipeline in background thread
    t = threading.Thread(
        target=_run_pipeline,
        args=(job_id, cmd, job_dir, q),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/progress/<job_id>")
def progress(job_id):
    if job_id not in JOBS:
        return jsonify({"error": "Job not found"}), 404

    def generate():
        q = JOBS[job_id]["log_queue"]
        while True:
            try:
                msg = q.get(timeout=30)
                yield f"data: {json.dumps(msg)}\n\n"
                if msg.get("type") in ("done", "error"):
                    break
            except queue.Empty:
                # Keep-alive ping so browser doesn't close the connection
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/download/<job_id>/<filename>")
def download(job_id, filename):
    if job_id not in JOBS:
        return "Job not found", 404

    file_path = Path(JOBS[job_id]["output_dir"]) / filename
    if not file_path.exists():
        return "File not found", 404

    return send_file(str(file_path), as_attachment=True, download_name=filename)


# ── Pipeline runner ───────────────────────────────────────────────────────────

def _run_pipeline(job_id: str, cmd: list, job_dir: Path, q: queue.Queue):
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        for line in proc.stdout:
            line = line.rstrip()
            if line:
                q.put({"type": "log", "text": line})

        proc.wait()

        if proc.returncode != 0:
            q.put({"type": "error", "text": f"Pipeline exited with code {proc.returncode}"})
            JOBS[job_id]["status"] = "error"
            return

        # Collect output files and transcript
        files      = _collect_output_files(job_dir)
        transcript = ""
        txt_path   = job_dir / "transcript.txt"
        if txt_path.exists():
            transcript = txt_path.read_text(encoding="utf-8")

        JOBS[job_id]["status"] = "done"
        q.put({
            "type":       "done",
            "job_id":     job_id,
            "files":      files,
            "transcript": transcript,
        })

    except Exception as e:
        q.put({"type": "error", "text": str(e)})
        JOBS[job_id]["status"] = "error"


def _collect_output_files(job_dir: Path) -> list:
    """Return list of downloadable output filenames."""
    wanted = {".wav", ".mp4", ".mkv", ".avi", ".mov", ".txt", ".json"}
    skip   = {"audio_raw.wav"}
    return [
        f.name
        for f in sorted(job_dir.iterdir())
        if f.suffix.lower() in wanted
        and f.name not in skip
        and not f.name.endswith(".log")
    ]


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 56)
    print("  VOICEFORGE — Speech Restoration UI")
    print("=" * 56)
    print("  Open in browser: http://localhost:5000")
    print("  Press Ctrl+C to stop")
    print("=" * 56 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
