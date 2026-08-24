import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare ISIC 2020 metadata for embedding and retrieval.")
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory of downloaded ISIC 2020 data",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for cleaned metadata CSV",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="metadata_clean.csv",
        help="Clean metadata CSV filename",
    )
    return parser.parse_args()


def normalize_path(path_str: str) -> Path:
    return Path(path_str.replace("\\", "/")).expanduser().resolve()


def find_image_dir(base_path: Path, split_name: str) -> Path:
    split_candidates = {
        "train": [
            base_path / "ISIC_2020_Dataset" / "train",
            base_path / "ISIC_2020_Dataset" / "ISIC_2020_Train",
        ],
        "test": [
            base_path / "ISIC_2020_Dataset" / "test",
            base_path / "ISIC_2020_Dataset" / "ISIC_2020_Test",
        ],
    }

    for candidate in split_candidates[split_name]:
        if candidate.is_dir():
            return candidate

    candidates = ", ".join(str(path) for path in split_candidates[split_name])
    raise FileNotFoundError(f"{split_name} image directory not found. Checked: {candidates}")


def image_filename(value: str) -> str:
    path = Path(str(value))
    return path.name if path.suffix else f"{path.name}.jpg"


def normalize_meta(df: pd.DataFrame, split_name: str, base_path: Path) -> pd.DataFrame:
    df = df.copy()
    lower_cols = {col.lower(): col for col in df.columns}

    if "image_name" in lower_cols:
        image_col = lower_cols["image_name"]
    elif "image_id" in lower_cols:
        image_col = lower_cols["image_id"]
    elif "image" in lower_cols:
        image_col = lower_cols["image"]
    else:
        raise ValueError("Metadata missing image_name, image_id, or image column.")

    if "benign_malignant" in lower_cols:
        label_col = lower_cols["benign_malignant"]
    elif "dx" in lower_cols:
        label_col = lower_cols["dx"]
    else:
        label_col = None

    df["image_id"] = df[image_col].astype(str).str.replace(".jpg", "", case=False)
    image_dir = find_image_dir(base_path, split_name)

    df["image_path"] = df[image_col].astype(str).apply(
        lambda x: str((image_dir / image_filename(x)).relative_to(base_path))
    )
    df["label"] = df[label_col].astype(str) if label_col is not None else ""
    df["target"] = df[lower_cols["target"]] if "target" in lower_cols else df["label"]
    df["split"] = split_name
    df["id"] = df["image_id"]
    df["path"] = df["image_path"]

    return df[["image_id", "image_path", "label", "target", "split", "id", "path"]]


def main():
    args = parse_args()
    data_root = normalize_path(args.data_root)
    out_dir = normalize_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_meta_path = data_root / "ISIC_2020_Dataset" / "ISIC_2020_Train_Metadata.csv"
    test_meta_path = data_root / "ISIC_2020_Dataset" / "ISIC_2020_Test_Metadata.csv"

    if not train_meta_path.is_file():
        raise FileNotFoundError(f"Train metadata file not found: {train_meta_path}")

    print(f"Loading train metadata from {train_meta_path}")
    train_df = pd.read_csv(train_meta_path)
    train_out = normalize_meta(train_df, "train", data_root)

    frames = [train_out]
    if test_meta_path.is_file():
        print(f"Loading test metadata from {test_meta_path}")
        test_df = pd.read_csv(test_meta_path)
        test_out = normalize_meta(test_df, "test", data_root)
        frames.append(test_out)
    else:
        print(f"Warning: test metadata not found at {test_meta_path}, skipping test split.")

    result = pd.concat(frames, ignore_index=True)
    output_path = out_dir / args.output_name
    result.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Saved cleaned metadata to {output_path}")
    print(f"Total records: {len(result)}")


if __name__ == '__main__':
    main()
