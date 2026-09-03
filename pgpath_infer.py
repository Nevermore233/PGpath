#!/usr/bin/env python3
"""Run PGpath branch-node inference for new population feature profiles.

The script loads a trained PGpath model, applies the training feature order and
normalization to new population k-mer features, and writes predicted branch-node
labels to a CSV file.

Two normalization modes are supported:
1. Full training feature matrix: fit StandardScaler from the original matrix.
2. Selected k-mer list plus scaler statistics: use pre-exported mean and scale.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


DEFAULT_HIDDEN_DIM = 1024
DEFAULT_BATCH_SIZE = 1024


class FastPangenomeNet(nn.Module):
    """Multi-task branch-node prediction network.

    The architecture must be identical to the model definition used during
    training.
    """

    def __init__(
        self,
        input_dim: int,
        num_branches: int,
        max_classes: int,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.num_branches = num_branches
        self.max_classes = max_classes

        self.extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )
        self.multi_task_head = nn.Linear(
            hidden_dim // 2,
            num_branches * max_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extractor(x)
        logits_flat = self.multi_task_head(features)
        return logits_flat.view(-1, self.num_branches, self.max_classes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PGpath inference and export predicted branch-node labels for "
            "new population samples."
        )
    )

    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Path to the trained PGpath model weights (.pth).",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Path to the new population k-mer feature CSV file.",
    )
    parser.add_argument(
        "-t",
        "--training-resource",
        required=True,
        help=(
            "Path to the original training feature CSV, or a selected k-mer list "
            "when --scaler-stats is provided."
        ),
    )
    parser.add_argument(
        "--scaler-stats",
        default=None,
        help=(
            "Optional scaler statistics CSV exported by pgpath_prepare_features.py. "
            "Required when -t points to a k-mer list instead of the full training feature matrix."
        ),
    )
    parser.add_argument(
        "-l",
        "--labels",
        required=True,
        help="Path to the original branch label CSV used to rebuild label mappings.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Output CSV path. If omitted, a timestamped file named "
            "predicted_pangenome_paths_YYYYMMDD_HHMMSS.csv is written to the "
            "current directory."
        ),
    )
    parser.add_argument(
        "--confidence-output",
        default=None,
        help=(
            "Optional output CSV path for branch-node prediction confidence scores. "
            "If omitted, confidence scores are not saved."
        ),
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=DEFAULT_HIDDEN_DIM,
        help="Hidden dimension of the model. Must match the training setting.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for inference.",
    )

    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    return torch.device(device_arg)


def require_file(path: str | os.PathLike[str], label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {resolved}")
    return resolved


def resolve_output_path(output: str | None) -> Path:
    if output is None:
        timestamp = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path.cwd() / f"predicted_pangenome_paths_{timestamp}.csv"

    output_path = Path(output).expanduser()
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".csv")
    if output_path.parent and str(output_path.parent) not in ("", "."):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path.resolve()


def read_csv_indexed(path: Path, label: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception as exc:
        raise ValueError(f"Failed to read {label} as a CSV file: {path}") from exc

    if df.empty:
        raise ValueError(f"{label} is empty: {path}")
    if df.index.has_duplicates:
        duplicates = df.index[df.index.duplicated()].astype(str).unique().tolist()
        preview = ", ".join(duplicates[:10])
        raise ValueError(f"{label} contains duplicated sample IDs: {preview}")
    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].astype(str).unique().tolist()
        preview = ", ".join(duplicates[:10])
        raise ValueError(f"{label} contains duplicated columns: {preview}")
    return df


def validate_numeric_matrix(df: pd.DataFrame, label: str) -> np.ndarray:
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    values = numeric_df.to_numpy(dtype=np.float32)

    if not np.isfinite(values).all():
        bad_count = int((~np.isfinite(values)).sum())
        raise ValueError(
            f"{label} contains {bad_count} non-numeric, NaN, or infinite values."
        )
    return values


def read_kmer_list(path: Path) -> List[str]:
    kmers: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            if "," in value or "\t" in value:
                fields = [field.strip() for field in re_split_simple(value) if field.strip()]
                if len(fields) != 1:
                    raise ValueError(
                        f"Invalid k-mer list format at {path}:{line_number}. "
                        "Expected one k-mer per line."
                    )
                value = fields[0]
            kmers.append(value)

    if not kmers:
        raise ValueError(f"No k-mers were found in: {path}")
    if len(set(kmers)) != len(kmers):
        raise ValueError("Duplicated k-mers were found in the k-mer list.")
    return kmers


def re_split_simple(value: str) -> List[str]:
    if "\t" in value:
        return value.split("\t")
    return value.split(",")


def read_scaler_stats(path: Path) -> Tuple[List[str], np.ndarray, np.ndarray]:
    try:
        stats_df = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to read scaler statistics CSV: {path}") from exc

    required_columns = {"kmer", "mean", "scale"}
    missing = required_columns.difference(stats_df.columns)
    if missing:
        raise ValueError(f"Scaler statistics CSV is missing columns: {sorted(missing)}")
    if stats_df.empty:
        raise ValueError(f"Scaler statistics CSV is empty: {path}")
    if stats_df["kmer"].duplicated().any():
        duplicated = stats_df.loc[stats_df["kmer"].duplicated(), "kmer"].head(10).tolist()
        raise ValueError(f"Duplicated k-mers were found in scaler statistics: {duplicated}")

    kmers = stats_df["kmer"].astype(str).tolist()
    mean = pd.to_numeric(stats_df["mean"], errors="coerce").to_numpy(dtype=np.float32)
    scale = pd.to_numeric(stats_df["scale"], errors="coerce").to_numpy(dtype=np.float32)

    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError("Scaler statistics contain non-numeric, NaN, or infinite values.")
    if np.any(scale == 0):
        raise ValueError("Scaler statistics contain zero scale values.")

    return kmers, mean, scale


def build_label_mapping(
    labels_df: pd.DataFrame,
) -> Tuple[List[str], int, int, Dict[int, Dict[int, str]], List[int]]:
    if labels_df.shape[1] == 0:
        raise ValueError("The label file must contain at least one branch column.")

    branch_names = labels_df.columns.tolist()
    num_branches = len(branch_names)
    max_classes = int(labels_df.nunique(dropna=True).max())

    if max_classes <= 0:
        raise ValueError("No valid branch-node labels were found in the label file.")

    idx_to_val_mapping: Dict[int, Dict[int, str]] = {}
    class_counts: List[int] = []
    for branch_idx, col in enumerate(branch_names):
        unique_nodes = labels_df[col].dropna().unique()
        if len(unique_nodes) == 0:
            raise ValueError(f"Branch '{col}' has no valid candidate node labels.")
        idx_to_val_mapping[branch_idx] = {
            class_idx: str(node_val) for class_idx, node_val in enumerate(unique_nodes)
        }
        class_counts.append(len(unique_nodes))

    return branch_names, num_branches, max_classes, idx_to_val_mapping, class_counts


def align_new_features(
    new_df: pd.DataFrame,
    feature_columns: List[str],
) -> pd.DataFrame:
    missing_columns = [col for col in feature_columns if col not in new_df.columns]
    if missing_columns:
        preview = ", ".join(missing_columns[:10])
        suffix = "..." if len(missing_columns) > 10 else ""
        raise ValueError(
            f"The input feature file is missing {len(missing_columns)} required "
            f"feature columns: {preview}{suffix}"
        )

    return new_df.loc[:, feature_columns]


def build_valid_class_mask(
    class_counts: Sequence[int],
    max_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Return a mask that suppresses padding classes for each branch."""
    if any(count <= 0 or count > max_classes for count in class_counts):
        raise ValueError("Invalid branch candidate-class counts in the label file.")

    mask = torch.zeros((len(class_counts), max_classes), dtype=torch.bool, device=device)
    for branch_idx, class_count in enumerate(class_counts):
        mask[branch_idx, :class_count] = True
    return mask


def mask_invalid_class_logits(logits: torch.Tensor, valid_class_mask: torch.Tensor) -> torch.Tensor:
    """Prevent padded output classes from being selected for sparse branches."""
    if logits.ndim != 3 or valid_class_mask.shape != logits.shape[1:]:
        raise ValueError("The valid-class mask is incompatible with the model output shape.")
    minimum = torch.finfo(logits.dtype).min
    return logits.masked_fill(~valid_class_mask.unsqueeze(0), minimum)


def load_model_state(model_path: Path, device: torch.device) -> Mapping[str, torch.Tensor]:
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model weights: {model_path}") from exc

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    if not isinstance(checkpoint, Mapping):
        raise ValueError(
            "Unsupported model checkpoint format. Expected a state_dict or a "
            "dictionary containing 'model_state_dict'."
        )

    return checkpoint


def predict_in_batches(
    model: nn.Module,
    x_tensor: torch.Tensor,
    batch_size: int,
    valid_class_mask: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")

    predictions = []
    confidences = []

    model.eval()

    with torch.no_grad():
        for start in range(0, x_tensor.size(0), batch_size):
            batch = x_tensor[start : start + batch_size]

            logits = mask_invalid_class_logits(model(batch), valid_class_mask)

            probabilities = torch.softmax(logits, dim=2)

            batch_confidence, batch_preds = torch.max(
                probabilities,
                dim=2,
            )

            predictions.append(
                batch_preds.cpu().numpy()
            )

            confidences.append(
                batch_confidence.cpu().numpy()
            )

    return (
        np.concatenate(predictions, axis=0),
        np.concatenate(confidences, axis=0),
    )


def decode_predictions(
    preds: np.ndarray,
    sample_names: List[str],
    branch_names: List[str],
    idx_to_val_mapping: Dict[int, Dict[int, str]],
) -> pd.DataFrame:
    decoded_results = []

    for sample_idx, sample_name in enumerate(sample_names):
        sample_result = {"Sample_ID": sample_name}
        for branch_idx, branch_name in enumerate(branch_names):
            predicted_class_idx = int(preds[sample_idx, branch_idx])
            sample_result[branch_name] = idx_to_val_mapping[branch_idx].get(
                predicted_class_idx,
                "UNKNOWN",
            )
        decoded_results.append(sample_result)

    results_df = pd.DataFrame(decoded_results)
    results_df.set_index("Sample_ID", inplace=True)
    return results_df


def decode_confidences(
    confidences: np.ndarray,
    sample_names: List[str],
    branch_names: List[str],
) -> pd.DataFrame:
    confidence_results = []

    for sample_idx, sample_name in enumerate(sample_names):
        sample_result = {
            "Sample_ID": sample_name
        }

        for branch_idx, branch_name in enumerate(branch_names):
            sample_result[branch_name] = float(
                confidences[sample_idx, branch_idx]
            )

        confidence_results.append(sample_result)

    confidence_df = pd.DataFrame(confidence_results)
    confidence_df.set_index("Sample_ID", inplace=True)

    return confidence_df


def load_feature_columns_and_scale_new_data(
    training_resource_path: Path,
    scaler_stats_path: Path | None,
    new_df: pd.DataFrame,
) -> Tuple[np.ndarray, List[str]]:
    if scaler_stats_path is not None:
        kmers_from_scaler, mean, scale = read_scaler_stats(scaler_stats_path)
        kmers_from_resource = read_kmer_list(training_resource_path)
        if kmers_from_resource != kmers_from_scaler:
            raise ValueError(
                "The k-mer list and scaler statistics have different k-mer order or content."
            )

        aligned_new_df = align_new_features(new_df, kmers_from_scaler)
        new_values = validate_numeric_matrix(aligned_new_df, "Input feature file")
        x_new_scaled = (new_values - mean.reshape(1, -1)) / scale.reshape(1, -1)
        if not np.isfinite(x_new_scaled).all():
            raise ValueError("Feature scaling produced NaN or infinite values.")
        return x_new_scaled.astype(np.float32), kmers_from_scaler

    train_features_df = read_csv_indexed(training_resource_path, "Training feature file")
    train_values = validate_numeric_matrix(train_features_df, "Training feature file")
    feature_columns = [str(col) for col in train_features_df.columns.tolist()]

    aligned_new_df = align_new_features(new_df, feature_columns)
    new_values = validate_numeric_matrix(aligned_new_df, "Input feature file")

    scaler = StandardScaler()
    scaler.fit(train_values)
    x_new_scaled = scaler.transform(new_values)
    if not np.isfinite(x_new_scaled).all():
        raise ValueError("Feature scaling produced NaN or infinite values.")
    return x_new_scaled.astype(np.float32), feature_columns


def atomic_to_csv(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Write a CSV atomically so failed writes do not leave a partial output."""
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".csv.tmp",
            prefix=f".{output_path.stem}.",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            temp_name = handle.name
            dataframe.to_csv(handle)
        os.replace(temp_name, output_path)
        temp_name = None
    finally:
        if temp_name is not None:
            try:
                Path(temp_name).unlink(missing_ok=True)
            except OSError:
                pass


def run_inference(args: argparse.Namespace) -> Path:
    model_path = require_file(args.model, "Model file")
    new_data_path = require_file(args.input, "Input feature file")
    training_resource_path = require_file(args.training_resource, "Training resource file")
    scaler_stats_path = (
        require_file(args.scaler_stats, "Scaler statistics file")
        if args.scaler_stats is not None
        else None
    )
    labels_path = require_file(args.labels, "Label file")
    output_path = resolve_output_path(args.output)
    device = resolve_device(args.device)
    if args.hidden_dim < 2:
        raise ValueError("hidden_dim must be at least 2.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    print(f"Device: {device}")
    print("Loading model inputs...")

    labels_df = read_csv_indexed(labels_path, "Label file")
    (
        branch_names,
        num_branches,
        max_classes,
        idx_to_val_mapping,
        class_counts,
    ) = build_label_mapping(labels_df)

    new_df = read_csv_indexed(new_data_path, "Input feature file")
    sample_names = [str(name) for name in new_df.index.tolist()]
    x_new_scaled, feature_columns = load_feature_columns_and_scale_new_data(
        training_resource_path=training_resource_path,
        scaler_stats_path=scaler_stats_path,
        new_df=new_df,
    )
    x_new = torch.as_tensor(x_new_scaled, dtype=torch.float32, device=device)

    model = FastPangenomeNet(
        input_dim=len(feature_columns),
        num_branches=num_branches,
        max_classes=max_classes,
        hidden_dim=args.hidden_dim,
    ).to(device)
    valid_class_mask = build_valid_class_mask(class_counts, max_classes, device)

    state_dict = load_model_state(model_path, device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise RuntimeError(
            "Model weights are incompatible with the current architecture. Check "
            "--hidden-dim, the feature resource, scaler statistics, and the label file."
        ) from exc

    print(
        f"Running inference for {len(sample_names)} sample(s), "
        f"{num_branches} branches, {len(feature_columns)} k-mer features, "
        f"and up to {max_classes} classes per branch."
    )

    preds, confidences = predict_in_batches(
        model,
        x_new,
        args.batch_size,
        valid_class_mask,
    )
    results_df = decode_predictions(
        preds,
        sample_names,
        branch_names,
        idx_to_val_mapping,
    )

    atomic_to_csv(results_df, output_path)
    print(f"Saved predictions: {output_path}")

    if args.confidence_output is not None:
        confidence_df = decode_confidences(
            confidences,
            sample_names,
            branch_names,
        )

        confidence_output_path = resolve_output_path(args.confidence_output)
        atomic_to_csv(confidence_df, confidence_output_path)

        print(
            f"Saved confidence scores: {confidence_output_path}"
        )

    return output_path


def main() -> None:
    args = parse_args()
    try:
        run_inference(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
