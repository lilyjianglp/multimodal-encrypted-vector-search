import argparse
import csv
import random
from collections import Counter
from pathlib import Path


LABELS = {
    (0, 0): "normal",
    (1, 0): "crackles",
    (0, 1): "wheezes",
    (1, 1): "both",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare patient-disjoint ICBHI 2017 respiratory-cycle metadata."
    )
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--query-patients", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    audio_root = args.audio_root.resolve()
    out_dir = args.out_dir.resolve()

    wav_paths = sorted(audio_root.glob("*.wav"))
    txt_paths = sorted(audio_root.glob("*.txt"))
    if not wav_paths or not txt_paths:
        raise FileNotFoundError(f"No WAV/TXT files found in {audio_root}")

    patients = sorted({path.stem.split("_")[0] for path in wav_paths})
    if not 0 < args.query_patients < len(patients):
        raise ValueError(
            f"--query-patients must be between 1 and {len(patients) - 1}"
        )

    random.Random(args.seed).shuffle(patients)
    query_patients = set(patients[: args.query_patients])
    rows = []

    for annotation_path in txt_paths:
        recording_id = annotation_path.stem
        patient_id = recording_id.split("_")[0]
        wav_path = audio_root / f"{recording_id}.wav"
        if not wav_path.is_file():
            print(f"[WARN] Missing WAV for {annotation_path.name}")
            continue

        split = "query" if patient_id in query_patients else "corpus"
        for cycle_index, line in enumerate(
            annotation_path.read_text(errors="ignore").splitlines()
        ):
            parts = line.split()
            if len(parts) < 4:
                continue

            start_sec, end_sec = float(parts[0]), float(parts[1])
            crackles, wheezes = int(parts[2]), int(parts[3])
            if end_sec <= start_sec:
                print(f"[WARN] Invalid interval in {annotation_path.name}: {line}")
                continue

            rows.append(
                {
                    "cycle_id": f"{recording_id}_cycle_{cycle_index:03d}",
                    "recording_id": recording_id,
                    "patient_id": patient_id,
                    "wav_path": str(wav_path),
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "duration_sec": end_sec - start_sec,
                    "crackles": crackles,
                    "wheezes": wheezes,
                    "label": LABELS[(crackles, wheezes)],
                    "split": split,
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "metadata_cycles.csv"
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    split_counts = Counter(row["split"] for row in rows)
    label_counts = Counter((row["split"], row["label"]) for row in rows)
    corpus_patients = {row["patient_id"] for row in rows if row["split"] == "corpus"}
    actual_query_patients = {
        row["patient_id"] for row in rows if row["split"] == "query"
    }

    print(f"Saved: {output_path}")
    print(f"Recordings: {len(wav_paths)}")
    print(f"Cycles: {len(rows)}")
    print(f"Corpus cycles: {split_counts['corpus']}")
    print(f"Query cycles: {split_counts['query']}")
    print(f"Corpus patients: {len(corpus_patients)}")
    print(f"Query patients: {len(actual_query_patients)}")
    print(f"Patient overlap: {corpus_patients & actual_query_patients}")
    print("Label counts:")
    for (split, label), count in sorted(label_counts.items()):
        print(f"  {split:6s} {label:8s}: {count}")


if __name__ == "__main__":
    main()
