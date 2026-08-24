import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


HOME = Path.home()
CKKS_DIR = HOME / "Desktop/backend/ckks"
INDEX_DIR = HOME / "Desktop/backend/services/index"
ARTIFACT_DIR = HOME / "private-vector-search/artifacts/image"
BACKEND_DATA_DIR = HOME / "private-vector-search/backend_data/data_image"
SLOT_IDS_PATH = HOME / "slot_ids.json"
SCORE_DIR = CKKS_DIR / "scores_raw_raw"
CHUNK_PATH = Path("/tmp/image_candidate_ids_4096.json")
QUERY_PATH = Path("/tmp/image_query.npy")
CLUSTERS_PATH = Path("/tmp/image_clusters.txt")


def run(command, *, cwd=None, env=None):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the verified single-pack image CKKS retrieval pipeline."
    )
    parser.add_argument("image", type=Path, help="Query image path on the VM")
    parser.add_argument("--top-t", type=int, default=16, help="Number of real IVF clusters")
    parser.add_argument("--l", type=int, default=16, help="Cluster request length")
    parser.add_argument("--top-k", type=int, default=8, help="Number of results to print")
    parser.add_argument(
        "--query-out",
        type=Path,
        default=QUERY_PATH,
        help="Temporary normalized query vector path",
    )
    return parser.parse_args()


def normalize_query(path):
    query = np.load(path).astype("float32").reshape(-1)
    norm = float(np.linalg.norm(query))
    if norm <= 1e-12:
        raise RuntimeError("Query embedding has zero norm.")
    query /= norm
    np.save(path, query)
    print(f"[OK] query shape={query.shape}, norm={np.linalg.norm(query):.8f}")


def write_first_pack():
    ids = json.load(open(SLOT_IDS_PATH, encoding="utf-8"))
    if len(ids) < 4096:
        ids.extend([0] * (4096 - len(ids)))
    first_pack = [int(value) for value in ids[:4096]]

    valid = [value for value in first_pack if value != 0]
    if not valid:
        raise RuntimeError("The first candidate pack contains no valid IDs.")
    if min(valid) < 0 or max(valid) >= 44108:
        raise RuntimeError(
            f"Image candidate ID out of range: min={min(valid)}, max={max(valid)}"
        )

    json.dump(first_pack, open(CHUNK_PATH, "w", encoding="utf-8"))
    print(
        f"[OK] first pack: slots={len(first_pack)}, nonzero={len(valid)}, "
        f"range=({min(valid)}, {max(valid)})"
    )
    return first_pack


def build_d0512():
    d0512 = CKKS_DIR / "image/D0512"
    d0512.mkdir(parents=True, exist_ok=True)
    for path in d0512.iterdir():
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)

    run(
        [
            str(CKKS_DIR / "make_dia1_768"),
            "--context",
            str(CKKS_DIR / "context.seal"),
            "--ids",
            str(CHUNK_PATH),
            "--dim",
            "512",
            "--outdir",
            str(d0512),
            "--mode",
            "from-npy",
            "--npy",
            str(ARTIFACT_DIR / "embeddings/image_corpus_l2.npy"),
        ],
        cwd=CKKS_DIR,
    )


def run_search():
    clusters = CLUSTERS_PATH.read_text(encoding="utf-8").strip()
    env = os.environ.copy()
    env["CLUSTERS"] = clusters
    run([sys.executable, "search_real_ct.py", "--mode", "image"], cwd=CKKS_DIR, env=env)


def clear_scores():
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    for path in SCORE_DIR.glob("scores_*"):
        path.unlink()


def load_image_mapping():
    mapping_path = ARTIFACT_DIR / "embeddings/id_mapping.json"
    return json.load(open(mapping_path, encoding="utf-8"))


def load_metadata():
    metadata_path = ARTIFACT_DIR / "metadata_clean.csv"
    if not metadata_path.is_file():
        return {}

    result = {}
    with open(metadata_path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            result[row["image_id"]] = row
    return result


def collect_results(first_pack, top_k):
    csv_paths = sorted(SCORE_DIR.glob("scores_*.csv"))
    if len(csv_paths) != 4:
        raise RuntimeError(f"Expected 4 decrypted score CSV files, got {len(csv_paths)}.")

    parts = [
        np.loadtxt(path, delimiter=",", skiprows=1, usecols=1) for path in csv_paths
    ]
    scores = np.sum(parts, axis=0) * 512.0

    # The verified layout stores one valid candidate score every 512 slots.
    valid_slots = np.arange(0, 4096, 512)
    ranked = [
        (first_pack[position], float(scores[position]))
        for position in valid_slots
        if first_pack[position] != 0
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked[:top_k]


def main():
    args = parse_args()
    image = args.image.expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(image)

    run(
        [
            sys.executable,
            "embed_image.py",
            str(image),
            "--pca",
            str(ARTIFACT_DIR / "embeddings/image_pca_2048_to_512.pkl"),
            "--weight",
            str(CKKS_DIR / "image/resnet50-0676ba61.pth"),
            "--out",
            str(args.query_out),
        ],
        cwd=CKKS_DIR,
    )
    normalize_query(args.query_out)

    run(
        [
            str(CKKS_DIR / "ckks_make_ctq_from_npy"),
            "--context",
            str(CKKS_DIR / "context.seal"),
            "--pk",
            str(CKKS_DIR / "pk.bin"),
            "--npy",
            str(args.query_out),
            "--dim",
            "512",
            "--out",
            str(CKKS_DIR / "ct_q.bin"),
        ],
        cwd=CKKS_DIR,
    )

    run(
        [
            sys.executable,
            "client_pick_clusters.py",
            "--q",
            str(args.query_out),
            "--centers",
            str(BACKEND_DATA_DIR / "centers.npy"),
            "--topT",
            str(args.top_t),
            "--L",
            str(args.l),
            "--session",
            "image-cli",
            "--metric",
            "cos",
            "--out",
            str(CLUSTERS_PATH),
        ],
        cwd=CKKS_DIR,
    )

    # First request refreshes /home/wen/slot_ids.json for this query.
    run_search()
    first_pack = write_first_pack()
    build_d0512()

    clear_scores()
    run_search()
    run(
        [
            str(CKKS_DIR / "ckks_decrypt_dump"),
            "--context",
            str(CKKS_DIR / "context.seal"),
            "--sk",
            str(CKKS_DIR / "sk.bin"),
            "--scores_dir",
            str(SCORE_DIR),
            "--dim",
            "512",
        ],
        cwd=CKKS_DIR,
    )

    mapping = load_image_mapping()
    metadata = load_metadata()
    results = collect_results(first_pack, args.top_k)

    print("\nrank\trow_id\timage_id\tscore\tpath")
    for rank, (row_id, score) in enumerate(results, 1):
        image_id = mapping.get(str(row_id), "<unknown>")
        image_path = metadata.get(image_id, {}).get("image_path", "")
        print(f"{rank}\t{row_id}\t{image_id}\t{score:.10f}\t{image_path}")


if __name__ == "__main__":
    main()
