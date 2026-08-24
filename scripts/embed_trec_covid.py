import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return (x / norm).astype("float32")


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.no_grad()
def encode_texts(texts, tokenizer, model, device, batch_size=32, max_length=256):
    model.eval()
    outputs = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
        batch = texts[i:i + batch_size]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)

        emb = mean_pool(out.last_hidden_state, enc["attention_mask"])
        emb = emb.detach().cpu().numpy().astype("float32")
        outputs.append(emb)

    return np.vstack(outputs).astype("float32")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-csv", required=True)
    parser.add_argument("--queries-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--model-name",
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--limit", type=int, default=0, help="debug limit for corpus; 0 means full")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print("device:", device)
    print("model:", args.model_name)
    print("batch size:", args.batch_size)
    print("max length:", args.max_length)

    corpus_df = pd.read_csv(args.corpus_csv)
    queries_df = pd.read_csv(args.queries_csv)

    if args.limit and args.limit > 0:
        corpus_df = corpus_df.iloc[:args.limit].copy()

    corpus_texts = corpus_df["full_text"].fillna("").astype(str).tolist()
    query_texts = queries_df["query_text"].fillna("").astype(str).tolist()

    print("num corpus:", len(corpus_texts))
    print("num queries:", len(query_texts))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)

    corpus_768 = encode_texts(
        corpus_texts,
        tokenizer,
        model,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    queries_768 = encode_texts(
        query_texts,
        tokenizer,
        model,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    print("corpus_768:", corpus_768.shape)
    print("queries_768:", queries_768.shape)

    np.save(out_dir / "text_corpus_768.npy", corpus_768)
    np.save(out_dir / "text_queries_768.npy", queries_768)
    corpus_768.tofile(out_dir / "text_corpus_768.f32")
    queries_768.tofile(out_dir / "text_queries_768.f32")

    print("Fitting PCA 768 -> 512")
    pca = PCA(n_components=512, random_state=42)
    corpus_512 = pca.fit_transform(corpus_768).astype("float32")
    queries_512 = pca.transform(queries_768).astype("float32")

    joblib.dump(pca, out_dir / "text_pca_768_to_512.pkl")

    corpus_l2 = l2_normalize(corpus_512)
    queries_l2 = l2_normalize(queries_512)

    np.save(out_dir / "text_corpus_l2.npy", corpus_l2)
    np.save(out_dir / "text_queries_l2.npy", queries_l2)
    corpus_l2.tofile(out_dir / "text_corpus_l2.f32")
    queries_l2.tofile(out_dir / "text_queries_l2.f32")

    id_map = {}
    for _, row in corpus_df.iterrows():
        local_id = int(row["local_id"])
        id_map[str(local_id)] = {
            "doc_id": str(row["doc_id"]),
            "title": str(row.get("title", "")),
        }

    with open(out_dir / "text_id_map.json", "w", encoding="utf-8") as f:
        json.dump(id_map, f, ensure_ascii=False, indent=2)

    print("saved to:", out_dir)

    print("check corpus_l2 norm:")
    norms = np.linalg.norm(corpus_l2, axis=1)
    print("mean:", norms.mean())
    print("min/max:", norms.min(), norms.max())

    print("check queries_l2 norm:")
    qnorms = np.linalg.norm(queries_l2, axis=1)
    print("mean:", qnorms.mean())
    print("min/max:", qnorms.min(), qnorms.max())


if __name__ == "__main__":
    main()
