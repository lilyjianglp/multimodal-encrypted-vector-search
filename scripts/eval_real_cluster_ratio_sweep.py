#!/usr/bin/env python3
import argparse
import json
import subprocess
import time
from pathlib import Path


HOME = Path("/home/wen")
ROOT = HOME / "private-vector-search"
CKKS_ROOT = HOME / "Desktop/backend/ckks"
SCRIPT = ROOT / "scripts/eval_final_ckks_recall_batched.py"
OUT_ROOT = ROOT / "results/cluster_ratio_sweep"


def run(command):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, check=True)


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Run real CKKS final-recall experiments while varying real/fake cluster ratio."
    )
    parser.add_argument("--mode", choices=["image", "text", "audio"], required=True)
    parser.add_argument("--top-t-list", default="4,8,16,32")
    parser.add_argument("--l", type=int, default=32)
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--num-queries", type=int, default=1)
    parser.add_argument("--final-k", type=int, default=16)
    parser.add_argument("--rerank-k", type=int, default=100)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()

    top_t_values = parse_int_list(args.top_t_list)
    for top_t in top_t_values:
        if top_t > args.l:
            raise ValueError(f"top_t={top_t} cannot exceed l={args.l}")
        if args.l % top_t != 0 and top_t != args.l:
            print(f"[WARN] top_t={top_t} is not a divisor of L={args.l}; continuing.", flush=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Eval keys are adapter-local state. Upload once before the sweep, then skip
    # repeated uploads inside each scoring batch.
    run(["python3", str(CKKS_ROOT / "upload_evalkeys_v1.py")])

    rows = []
    for top_t in top_t_values:
        print(f"\n===== {args.mode} topT={top_t} L={args.l} =====", flush=True)
        started = time.time()
        run(
            [
                "python3",
                str(SCRIPT),
                "--mode",
                args.mode,
                "--query-index",
                str(args.query_index),
                "--num-queries",
                str(args.num_queries),
                "--final-k",
                str(args.final_k),
                "--rerank-k",
                str(args.rerank_k),
                "--candidate-k",
                str(args.candidate_k),
                "--top-t",
                str(top_t),
                "--l",
                str(args.l),
                "--timeout",
                str(args.timeout),
                "--skip-eval-keys",
            ]
        )
        elapsed = time.time() - started

        summary_path = (
            ROOT
            / "results/final_ckks_recall_batched"
            / args.mode
            / f"summary_final{args.final_k}_rerank{args.rerank_k}_topT{top_t}_L{args.l}.json"
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {
            "mode": args.mode,
            "top_t": top_t,
            "l": args.l,
            "fake_cluster_count": args.l - top_t,
            "real_cluster_ratio": top_t / args.l,
            "final_k": args.final_k,
            "rerank_k": args.rerank_k,
            "candidate_k": args.candidate_k,
            "num_queries": args.num_queries,
            "mean_final_recall_at_k": summary["mean_final_recall_at_k"],
            "mean_rerank_candidate_recall_at_k": summary["mean_rerank_candidate_recall_at_k"],
            "ranking_consistent_rate": summary["ranking_consistent_rate"],
            "mean_abs_error": summary["mean_abs_error"],
            "mean_max_abs_error": summary["mean_max_abs_error"],
            "elapsed_sec": elapsed,
            "summary_path": str(summary_path),
        }
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    out_path = (
        OUT_ROOT
        / f"{args.mode}_final{args.final_k}_rerank{args.rerank_k}_L{args.l}_q{args.num_queries}.json"
    )
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\n===== sweep saved =====")
    print(out_path)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
