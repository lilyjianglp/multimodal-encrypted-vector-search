import argparse
import csv
import json
from pathlib import Path

import librosa
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract patient-disjoint ICBHI cycle-level Log-Mel features."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--min-seconds", type=float, default=0.5)
    return parser.parse_args()


def extract_logmel_stats(segment, sample_rate):
    mel = librosa.feature.melspectrogram(
        y=segment,
        sr=sample_rate,
        n_fft=1024,
        hop_length=160,
        win_length=400,
        n_mels=64,
        fmin=50,
        fmax=min(2000, sample_rate // 2),
        power=2.0,
    )
    logmel = librosa.power_to_db(mel + 1e-10, ref=np.max)
    stats = [
        np.mean(logmel, axis=1),
        np.std(logmel, axis=1),
        np.min(logmel, axis=1),
        np.max(logmel, axis=1),
        np.percentile(logmel, 10, axis=1),
        np.percentile(logmel, 25, axis=1),
        np.percentile(logmel, 75, axis=1),
        np.percentile(logmel, 90, axis=1),
    ]
    feature = np.concatenate(stats).astype("float32")
    if feature.shape != (512,):
        raise RuntimeError(f"Expected a 512-d feature, got {feature.shape}")
    return feature


def l2_normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def load_metadata(path):
    with path.open(newline="", encoding="utf-8") as metadata_file:
        rows = list(csv.DictReader(metadata_file))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def save_split(out_dir, split, features, rows):
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    np.save(split_dir / f"audio_{split}_l2.npy", features)
    features.tofile(split_dir / f"audio_{split}_l2.f32")

    id_map = {}
    with (split_dir / "ids.txt").open("w", encoding="utf-8") as ids_file:
        for row_id, row in enumerate(rows):
            item = {
                "row_id": row_id,
                "cycle_id": row["cycle_id"],
                "recording_id": row["recording_id"],
                "patient_id": row["patient_id"],
                "label": row["label"],
                "wav_path": row["wav_path"],
                "start_sec": float(row["start_sec"]),
                "end_sec": float(row["end_sec"]),
            }
            id_map[str(row_id)] = item
            ids_file.write(f"{row['cycle_id']}\n")

    with (split_dir / "id_mapping.json").open("w", encoding="utf-8") as map_file:
        json.dump(id_map, map_file, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    rows = load_metadata(args.metadata.resolve())
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    min_length = int(args.min_seconds * args.sample_rate)
    recording_cache = {}
    raw_features = []

    for index, row in enumerate(rows, start=1):
        wav_path = Path(row["wav_path"])
        cache_key = str(wav_path)
        if cache_key not in recording_cache:
            audio, _ = librosa.load(
                wav_path, sr=args.sample_rate, mono=True, dtype=np.float32
            )
            recording_cache[cache_key] = audio

        audio = recording_cache[cache_key]
        start = max(0, int(float(row["start_sec"]) * args.sample_rate))
        end = min(len(audio), int(float(row["end_sec"]) * args.sample_rate))
        segment = np.asarray(audio[start:end], dtype="float32")
        if len(segment) < min_length:
            segment = np.pad(segment, (0, min_length - len(segment)))

        peak = float(np.max(np.abs(segment))) if len(segment) else 0.0
        if peak > 0:
            segment = segment / peak

        raw_features.append(extract_logmel_stats(segment, args.sample_rate))
        if index % 250 == 0 or index == len(rows):
            print(f"Extracted {index}/{len(rows)} cycles", flush=True)

    raw_features = np.vstack(raw_features).astype("float32")
    corpus_mask = np.asarray([row["split"] == "corpus" for row in rows])
    query_mask = np.asarray([row["split"] == "query" for row in rows])

    # Fit preprocessing on the retrieval corpus only to prevent query leakage.
    mean = raw_features[corpus_mask].mean(axis=0, keepdims=True)
    std = raw_features[corpus_mask].std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)

    standardized = (raw_features - mean) / std
    normalized = l2_normalize(standardized).astype("float32")

    corpus_features = normalized[corpus_mask]
    query_features = normalized[query_mask]
    corpus_rows = [row for row in rows if row["split"] == "corpus"]
    query_rows = [row for row in rows if row["split"] == "query"]

    np.save(out_dir / "audio_feature_mean.npy", mean.astype("float32"))
    np.save(out_dir / "audio_feature_std.npy", std.astype("float32"))
    save_split(out_dir, "corpus", corpus_features, corpus_rows)
    save_split(out_dir, "query", query_features, query_rows)

    print(f"Saved: {out_dir}")
    print(f"Corpus shape: {corpus_features.shape}")
    print(f"Query shape: {query_features.shape}")
    print(
        "Corpus norm mean/std:",
        float(np.linalg.norm(corpus_features, axis=1).mean()),
        float(np.linalg.norm(corpus_features, axis=1).std()),
    )
    print(
        "Query norm mean/std:",
        float(np.linalg.norm(query_features, axis=1).mean()),
        float(np.linalg.norm(query_features, axis=1).std()),
    )


if __name__ == "__main__":
    main()
