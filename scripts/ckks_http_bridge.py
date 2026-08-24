#!/usr/bin/env python3
import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_file


HOME = Path("/home/wen")
RUNNER = HOME / "private-vector-search/code/scripts/run_multimodal_ckks.py"
RESULTS_ROOT = HOME / "private-vector-search/results"

MODES = {"image", "text", "audio"}

app = Flask(__name__)


def result_dir(mode):
    return RESULTS_ROOT / f"{mode}_ckks/full_flow"


def metadata_id(mode, metadata):
    if mode == "image":
        return metadata.get("external_id") or str(metadata.get("row_id", "unknown"))
    if mode == "text":
        return metadata.get("doc_id") or metadata.get("title") or str(metadata.get("row_id", "unknown"))
    if mode == "audio":
        return metadata.get("cycle_id") or str(metadata.get("row_id", "unknown"))
    return str(metadata.get("row_id", "unknown"))


def load_topk(mode):
    rows_path = result_dir(mode) / "top8_results.csv"
    if not rows_path.exists():
        raise FileNotFoundError(f"missing result file: {rows_path}")

    topk = []
    with rows_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            metadata = {}
            raw_metadata = row.get("metadata") or "{}"
            try:
                metadata = json.loads(raw_metadata)
            except json.JSONDecodeError:
                metadata = {"raw": raw_metadata}

            item = {
                "rank": int(row["rank"]),
                "id": metadata_id(mode, metadata),
                "score": float(row["ckks_score"]),
                "plain_score": float(row["plain_score"]),
                "ckks_score": float(row["ckks_score"]),
                "abs_error": float(row["abs_error"]),
                "row_id": int(row["row_id"]),
                "metadata": metadata,
            }
            if mode == "text" and metadata.get("title"):
                item["title"] = metadata["title"]
            if mode == "audio":
                item["label"] = metadata.get("label")
                item["patient_id"] = metadata.get("patient_id")
            topk.append(item)
    return topk


def load_summary(mode):
    summary_path = result_dir(mode) / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return {}


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "runner": str(RUNNER)})


@app.route("/search", methods=["OPTIONS"])
def search_options():
    return ("", 204)


@app.route("/search", methods=["POST"])
def search():
    started = time.time()
    payload = request.get_json(force=True, silent=True) or {}
    mode = payload.get("type")
    query_path = payload.get("query_path")

    if mode not in MODES:
        return jsonify({"status": "error", "message": f"unsupported type: {mode}"}), 400
    if not query_path:
        return jsonify({"status": "error", "message": "missing query_path"}), 400

    query = Path(query_path)
    if query.suffix.lower() != ".npy":
        return jsonify({
            "status": "error",
            "message": "This bridge currently expects a prepared 512-d .npy query path. Use /tmp/query_image.npy, /tmp/query_text.npy, or /tmp/query_audio.npy.",
        }), 400

    command = ["python3", str(RUNNER), mode, "--query", str(query)]
    try:
        completed = subprocess.run(
            command,
            cwd=str(HOME / "Desktop/backend/ckks"),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=240,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return jsonify({"status": "error", "message": "CKKS workflow timed out"}), 504

    if completed.returncode != 0:
        return jsonify({
            "status": "error",
            "message": f"CKKS workflow failed with exit code {completed.returncode}",
            "logs": completed.stdout[-8000:],
        }), 500

    summary = load_summary(mode)
    topk = load_topk(mode)
    elapsed_ms = int((time.time() - started) * 1000)
    result = {
        "mode": mode,
        "topk": topk,
        "e2e_latency_ms": elapsed_ms,
        "ckks_summary": summary,
        "logs": completed.stdout[-8000:],
    }
    return jsonify({"status": "success", "result": result})


@app.route("/api/text-content/<path:file_id>", methods=["GET"])
def text_content(file_id):
    return jsonify({"id": file_id, "content": "Text preview is not wired yet; retrieval metadata is shown in the result list."})


@app.route("/api/files/<path:filename>", methods=["GET"])
def files(filename):
    path = Path(filename)
    if path.is_absolute() and path.exists():
        return send_file(path)
    return jsonify({"status": "error", "message": f"file preview not found: {filename}"}), 404


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
