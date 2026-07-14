#!/usr/bin/env python3
"""Extract PGpath selected k-mers and scaler statistics from a training feature matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract feature column names as selected k-mers and export StandardScaler "
            "statistics from the PGpath training feature matrix."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Training feature CSV. The first column is treated as the sample index.",
    )
    parser.add_argument(
        "-k",
        "--kmers-output",
        default="pgpath_selected_kmers.txt",
        help="Output text file containing one selected k-mer per line.",
    )
    parser.add_argument(
        "-s",
        "--scaler-output",
        default="pgpath_scaler_stats.csv",
        help="Output CSV containing k-mer, mean, and scale values for inference.",
    )
    return parser.parse_args()


def require_file(path: str, label: str) -> Path:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {file_path}")
    return file_path


def resolve_output_path(path: str) -> Path:
    output_path = Path(path).expanduser().resolve()
    if output_path.parent and not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def read_training_features(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception as exc:
        raise ValueError(f"Failed to read training feature CSV: {path}") from exc

    if df.empty:
        raise ValueError(f"Training feature CSV is empty: {path}")
    if df.shape[1] == 0:
        raise ValueError("Training feature CSV must contain at least one feature column.")
    if df.columns.duplicated().any():
        duplicated = df.columns[df.columns.duplicated()].tolist()[:10]
        raise ValueError(f"Duplicated feature columns were found: {duplicated}")

    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    values = numeric_df.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        bad_count = int((~np.isfinite(values)).sum())
        raise ValueError(
            f"Training feature CSV contains {bad_count} non-numeric, NaN, or infinite values."
        )

    return numeric_df


def write_kmer_list(kmers: list[str], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for kmer in kmers:
            handle.write(f"{kmer}\n")


def write_scaler_stats(df: pd.DataFrame, output_path: Path) -> None:
    scaler = StandardScaler()
    scaler.fit(df.to_numpy(dtype=np.float32))

    stats_df = pd.DataFrame(
        {
            "kmer": [str(col) for col in df.columns],
            "mean": scaler.mean_.astype(np.float64),
            "scale": scaler.scale_.astype(np.float64),
        }
    )
    stats_df.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()
    try:
        input_path = require_file(args.input, "Training feature CSV")
        kmers_output = resolve_output_path(args.kmers_output)
        scaler_output = resolve_output_path(args.scaler_output)

        train_df = read_training_features(input_path)
        kmers = [str(col) for col in train_df.columns]

        write_kmer_list(kmers, kmers_output)
        write_scaler_stats(train_df, scaler_output)

        print(f"Selected k-mers: {len(kmers):,}")
        print(f"Saved k-mer list: {kmers_output}")
        print(f"Saved scaler stats: {scaler_output}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
