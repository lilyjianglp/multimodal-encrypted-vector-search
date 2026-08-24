import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np


HOME = Path("/home/wen")
CKKS_ROOT = HOME / "Desktop/backend/ckks"
BACKEND_ROOT = HOME / "private-vector-search/backend_data"
RESULTS_ROOT = HOME / "private-vector-search/results"

MODALITIES = {
    "image": {
        "gateway": "127.0.0.1:50052",
        "top_t": 16,
        "centers": CKKS_ROOT / "image/centers.npy",
        "corpus": CKKS_ROOT / "image/corpus_l2.npy",
        "id_map": CKKS_ROOT / "image/id_mapping.json",
        "data_dir": BACKEND_ROOT / "data_image",
        "d0512": CKKS_ROOT / "image/D0512",
    },
    "text": {
        "gateway": "127.0.0.1:50058",
        "top_t": 32,
        "centers": CKKS_ROOT / "text/centers.npy",
        "corpus": CKKS_ROOT / "text/corpus_l2.npy",
        "id_map": CKKS_ROOT / "text/id_mapping.json",
        "data_dir": BACKEND_ROOT / "data_text",
        "d0512": CKKS_ROOT / "text/D0512",
    },
    "audio": {
        "gateway": "127.0.0.1:50057",
        "top_t": 8,
        "centers": CKKS_ROOT / "audio/centers.npy",
        "corpus": CKKS_ROOT / "audio/corpus_l2.npy",
        "id_map": CKKS_ROOT / "audio/id_mapping.json",
        "data_dir": BACKEND_ROOT / "data_audio",
        "d0512": CKKS_ROOT / "audio/D0512",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the verified single-batch Top-8 CKKS workflow."
    )
    parser.add_argument("modality", choices=MODALITIES)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--top-t", type=int)
    parser.add_argument("--l", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--skip-eval-keys", action="store_true")
    return parser.parse_args()


def run(command, env=None):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=CKKS_ROOT, env=env, check=True)


def clear_directory(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for child in directory.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def ensure_diag_link(data_dir, target):
    link = data_dir / "diag_blocks"
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.exists():
        shutil.rmtree(link)
    link.symlink_to(target)


def normalize_query(path):
    query = np.load(path).astype("float32")
    if query.shape != (512,):
        raise ValueError(f"Expected query shape (512,), got {query.shape}")
    query /= max(float(np.linalg.norm(query)), 1e-12)
    np.save(path, query)
    return query


def make_top8_layout(corpus_path, query, layout_path):
    corpus = np.load(corpus_path, mmap_mode="r")
    scores = np.asarray(corpus @ query)
    top8 = np.argsort(-scores)[:8]

    layout = [0] * 4096
    for index, row_id in enumerate(top8):
        layout[index * 512] = int(row_id)

    layout_path.write_text(json.dumps(layout))
    return corpus, scores, top8, layout


def wait_for_raw(raw_dir, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        files = sorted(raw_dir.glob("scores_*.bin"))
        if len(files) >= 4:
            return files
        time.sleep(1)
    raise TimeoutError(f"Expected 4 RAW score files in {raw_dir}")


def load_item(id_map, row_id):
    item = id_map.get(str(row_id), row_id)
    if isinstance(item, dict):
        return item
    return {"external_id": item}


def save_results(modality, query_path, corpus, query, layout, config):
    raw_dir = CKKS_ROOT / "scores_raw_raw"
    parts = [
        np.loadtxt(path, delimiter=",", skiprows=1, usecols=1)
        for path in sorted(raw_dir.glob("scores_*.csv"))
    ]
    if len(parts) != 4:
        raise RuntimeError(f"Expected 4 decrypted CSV files, got {len(parts)}")

    ckks_scores = np.sum(parts, axis=0) * 512
    id_map = json.loads(config["id_map"].read_text())
    rows = []

    for rank, slot in enumerate(range(0, 4096, 512), start=1):
        row_id = int(layout[slot])
        plain = float(corpus[row_id] @ query)
        encrypted = float(ckks_scores[slot])
        rows.append(
            {
                "rank": rank,
                "slot": slot,
                "row_id": row_id,
                "plain_score": plain,
                "ckks_score": encrypted,
                "abs_error": abs(plain - encrypted),
                "metadata": json.dumps(
                    load_item(id_map, row_id), ensure_ascii=False, sort_keys=True
                ),
            }
        )

    plain_order = np.argsort(-np.asarray([row["plain_score"] for row in rows]))
    ckks_order = np.argsort(-np.asarray([row["ckks_score"] for row in rows]))

    output_dir = RESULTS_ROOT / f"{modality}_ckks/full_flow"
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "top8_results.csv").open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "modality": modality,
        "query_path": str(query_path),
        "corpus_vectors": int(len(corpus)),
        "dimension": 512,
        "top_k": 8,
        "max_abs_error": max(row["abs_error"] for row in rows),
        "mean_abs_error": float(np.mean([row["abs_error"] for row in rows])),
        "ranking_consistent": bool(np.array_equal(plain_order, ckks_order)),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    for row in rows:
        print(row)


def main():
    args = parse_args()
    config = MODALITIES[args.modality]
    query_path = args.query.resolve()
    top_t = args.top_t or config["top_t"]
    center_count = int(np.load(config["centers"], mmap_mode="r").shape[0])
    if top_t > center_count or args.l > center_count:
        raise ValueError(
            f"top_t/L must not exceed {center_count} centers for {args.modality}"
        )

    query = normalize_query(query_path)
    layout_path = Path(f"/tmp/{args.modality}_top8_layout.json")
    clusters_path = Path(f"/tmp/{args.modality}_clusters.txt")
    corpus, _, _, layout = make_top8_layout(config["corpus"], query, layout_path)

    clear_directory(config["d0512"])
    run(
        [
            str(CKKS_ROOT / "make_dia1_768"),
            "--context",
            str(CKKS_ROOT / "context.seal"),
            "--ids",
            str(layout_path),
            "--dim",
            "512",
            "--outdir",
            str(config["d0512"]),
            "--mode",
            "from-npy",
            "--npy",
            str(config["corpus"]),
        ]
    )
    ensure_diag_link(config["data_dir"], config["d0512"])

    run(
        [
            str(CKKS_ROOT / "ckks_make_ctq_from_npy"),
            "--context",
            str(CKKS_ROOT / "context.seal"),
            "--pk",
            str(CKKS_ROOT / "pk.bin"),
            "--npy",
            str(query_path),
            "--dim",
            "512",
            "--out",
            str(CKKS_ROOT / "ct_q.bin"),
        ]
    )
    run(
        [
            "python3",
            str(CKKS_ROOT / "client_pick_clusters.py"),
            "--q",
            str(query_path),
            "--centers",
            str(config["centers"]),
            "--topT",
            str(top_t),
            "--L",
            str(args.l),
            "--session",
            f"full-flow-{args.modality}",
            "--metric",
            "cos",
            "--out",
            str(clusters_path),
        ]
    )

    if not args.skip_eval_keys:
        run(["python3", str(CKKS_ROOT / "upload_evalkeys_v1.py")])

    raw_dir = CKKS_ROOT / "scores_raw_raw"
    clear_directory(raw_dir)
    env = os.environ.copy()
    env["CLUSTERS"] = clusters_path.read_text().strip()
    env["GATEWAY_ADDR"] = config["gateway"]
    run(["python3", str(CKKS_ROOT / "search_real_ct.py"), "--mode", args.modality], env)
    raw_files = wait_for_raw(raw_dir, args.timeout)
    print("RAW files:", [(path.name, path.stat().st_size) for path in raw_files])

    run(
        [
            str(CKKS_ROOT / "ckks_decrypt_dump"),
            "--context",
            str(CKKS_ROOT / "context.seal"),
            "--sk",
            str(CKKS_ROOT / "sk.bin"),
            "--scores_dir",
            str(raw_dir),
            "--dim",
            "512",
        ]
    )
    save_results(args.modality, query_path, corpus, query, layout, config)


if __name__ == "__main__":
    main()
