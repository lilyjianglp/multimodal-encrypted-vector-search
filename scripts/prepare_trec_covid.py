import argparse
import json
from pathlib import Path

import pandas as pd


def read_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def find_file(root: Path, name: str):
    matches = list(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Cannot find {name} under {root}")
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = find_file(data_root, "corpus.jsonl")
    queries_path = find_file(data_root, "queries.jsonl")

    qrels_candidates = list(data_root.rglob("test.tsv"))
    if not qrels_candidates:
        raise FileNotFoundError(f"Cannot find qrels/test.tsv under {data_root}")
    qrels_path = qrels_candidates[0]

    print("corpus:", corpus_path)
    print("queries:", queries_path)
    print("qrels:", qrels_path)

    corpus = read_jsonl(corpus_path)
    queries = read_jsonl(queries_path)

    corpus_rows = []
    doc_id_map = {}

    for i, item in enumerate(corpus):
        doc_id = str(item.get("_id", ""))
        title = str(item.get("title", "") or "")
        text = str(item.get("text", "") or "")
        full_text = (title + "\n" + text).strip()

        corpus_rows.append({
            "local_id": i,
            "doc_id": doc_id,
            "title": title,
            "text": text,
            "full_text": full_text,
        })
        doc_id_map[doc_id] = i

    query_rows = []
    query_id_map = {}

    for i, item in enumerate(queries):
        qid = str(item.get("_id", ""))
        text = str(item.get("text", "") or "")

        query_rows.append({
            "query_local_id": i,
            "query_id": qid,
            "query_text": text,
        })
        query_id_map[qid] = i

    corpus_df = pd.DataFrame(corpus_rows)
    queries_df = pd.DataFrame(query_rows)

    qrels = pd.read_csv(qrels_path, sep="\t")

    # BEIR qrels 通常是 query-id / corpus-id / score
    if len(qrels.columns) >= 3:
        qrels = qrels.iloc[:, :3]
        qrels.columns = ["query_id", "doc_id", "score"]
    else:
        raise ValueError(f"Unexpected qrels format: {list(qrels.columns)}")

    qrels["query_id"] = qrels["query_id"].astype(str)
    qrels["doc_id"] = qrels["doc_id"].astype(str)

    qrels["query_local_id"] = qrels["query_id"].map(query_id_map)
    qrels["doc_local_id"] = qrels["doc_id"].map(doc_id_map)

    before = len(qrels)
    qrels = qrels.dropna(subset=["query_local_id", "doc_local_id"]).copy()
    after = len(qrels)

    qrels["query_local_id"] = qrels["query_local_id"].astype(int)
    qrels["doc_local_id"] = qrels["doc_local_id"].astype(int)
    qrels["score"] = qrels["score"].astype(float)

    corpus_out = out_dir / "corpus_clean.csv"
    queries_out = out_dir / "queries_clean.csv"
    qrels_out = out_dir / "qrels_clean.csv"
    doc_map_out = out_dir / "doc_id_map.json"
    query_map_out = out_dir / "query_id_map.json"

    corpus_df.to_csv(corpus_out, index=False)
    queries_df.to_csv(queries_out, index=False)
    qrels.to_csv(qrels_out, index=False)

    with open(doc_map_out, "w", encoding="utf-8") as f:
        json.dump(doc_id_map, f, ensure_ascii=False, indent=2)

    with open(query_map_out, "w", encoding="utf-8") as f:
        json.dump(query_id_map, f, ensure_ascii=False, indent=2)

    print("saved corpus:", corpus_out)
    print("saved queries:", queries_out)
    print("saved qrels:", qrels_out)
    print("num corpus:", len(corpus_df))
    print("num queries:", len(queries_df))
    print("num qrels before/after:", before, after)

    print("\ncorpus head:")
    print(corpus_df.head())

    print("\nqueries head:")
    print(queries_df.head())

    print("\nqrels head:")
    print(qrels.head())


if __name__ == "__main__":
    main()
