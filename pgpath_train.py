"""Train a topology-aware multi-task model for pangenome branch-node prediction.

The script keeps the original training behavior while providing a command-line
interface, stricter input validation, and timestamped model saving.
"""

from __future__ import annotations

import argparse
import datetime
import os
import random
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


Array2D = np.ndarray


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Train FastPangenomeNet with a topology-aware loss for "
            "pangenome branch-node prediction."
        )
    )
    parser.add_argument(
        "-i",
        "--features",
        required=True,
        help="Path to the k-mer feature CSV file.",
    )
    parser.add_argument(
        "-l",
        "--labels",
        required=True,
        help="Path to the branch-label CSV file.",
    )
    parser.add_argument(
        "-r",
        "--relations",
        required=True,
        help="Path to the branch-node relation CSV file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Path for saving the trained model. If omitted, the model is saved as "
            "./pgpath_model_YYYYMMDD_HHMMSS.pth."
        ),
    )
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--hidden-dim", type=int, default=1024, help="Hidden dimension of the model.")
    parser.add_argument("--lambda-graph", type=float, default=1e-4, help="Weight of the topology-aware loss.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for Adam.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Held-out test ratio used in the train/test split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device to use. Default: auto.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    """Resolve the training device from a user argument."""
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def require_file(path: str | Path, description: str) -> Path:
    """Return a valid file path or raise a clear error."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"{description} not found: {file_path}")
    return file_path


def load_csv_with_index(path: Path, description: str) -> pd.DataFrame:
    """Load a CSV file whose first column stores row identifiers."""
    try:
        dataframe = pd.read_csv(path, index_col=0)
    except Exception as exc:  # pragma: no cover - preserves detailed pandas error context.
        raise RuntimeError(f"Failed to read {description}: {path}") from exc

    if dataframe.empty:
        raise ValueError(f"{description} is empty: {path}")
    return dataframe


def align_labels_to_features(features_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Align label rows to feature rows when both files use the same sample IDs."""
    if features_df.shape[0] != labels_df.shape[0]:
        raise ValueError(
            "The feature and label files contain different numbers of samples: "
            f"features={features_df.shape[0]}, labels={labels_df.shape[0]}."
        )

    if features_df.index.equals(labels_df.index):
        return labels_df

    if set(features_df.index) == set(labels_df.index):
        print("Label rows were reordered to match the feature sample IDs.")
        return labels_df.loc[features_df.index]

    print("Warning: feature and label sample IDs do not match; row order will be used.")
    return labels_df


def encode_branch_labels(
    labels_df: pd.DataFrame,
) -> Tuple[Array2D, dict[str, int], int, int, int]:
    """Encode branch-node labels and build node-to-flat-index mapping."""
    num_samples = labels_df.shape[0]
    num_branches = labels_df.shape[1]

    if num_branches == 0:
        raise ValueError("The label file does not contain any branch columns.")

    max_classes_raw = labels_df.nunique(dropna=True).max()
    if pd.isna(max_classes_raw) or int(max_classes_raw) <= 0:
        raise ValueError("No valid branch-node labels were found in the label file.")
    max_classes = int(max_classes_raw)

    y_encoded = np.zeros((num_samples, num_branches), dtype=np.int64)
    node_to_flat_idx: dict[str, int] = {}

    for branch_idx, column in enumerate(labels_df.columns):
        unique_nodes = labels_df[column].dropna().unique()
        value_to_index = {value: class_idx for class_idx, value in enumerate(unique_nodes)}

        encoded_column = labels_df[column].map(value_to_index).fillna(-100).to_numpy(dtype=np.int64)
        y_encoded[:, branch_idx] = encoded_column

        for node_value, class_idx in value_to_index.items():
            flat_idx = branch_idx * max_classes + class_idx
            node_key = str(node_value)
            node_to_flat_idx[node_key] = flat_idx

    return y_encoded, node_to_flat_idx, num_samples, num_branches, max_classes


def build_laplacian(
    relations_path: Path,
    node_to_flat_idx: dict[str, int],
    num_branches: int,
    max_classes: int,
) -> torch.Tensor:
    """Build an unnormalized sparse graph Laplacian from branch-node relations."""
    relations_df = pd.read_csv(relations_path)
    required_columns = {"source", "target"}
    missing_columns = required_columns.difference(relations_df.columns)
    if missing_columns:
        raise ValueError(
            f"The relation file must contain columns {sorted(required_columns)}; "
            f"missing columns: {sorted(missing_columns)}."
        )

    n_total = num_branches * max_classes
    row_indices: list[int] = []
    col_indices: list[int] = []
    used_edges = 0

    for _, relation in relations_df.iterrows():
        source = str(relation["source"])
        target = str(relation["target"])

        if source not in node_to_flat_idx or target not in node_to_flat_idx:
            continue

        u = node_to_flat_idx[source]
        v = node_to_flat_idx[target]
        row_indices.extend([u, v])
        col_indices.extend([v, u])
        used_edges += 1

    edge_values = np.ones(len(row_indices), dtype=np.float32)
    adjacency = sp.coo_matrix((edge_values, (row_indices, col_indices)), shape=(n_total, n_total))
    adjacency.sum_duplicates()

    degree_values = np.asarray(adjacency.sum(axis=1)).ravel()
    laplacian = (sp.diags(degree_values) - adjacency).tocoo()

    indices = torch.from_numpy(np.vstack((laplacian.row, laplacian.col))).long()
    values = torch.from_numpy(laplacian.data).float()
    laplacian_tensor = torch.sparse_coo_tensor(indices, values, torch.Size(laplacian.shape)).coalesce()

    print(f"Topology relations used: {used_edges}/{len(relations_df)}")
    return laplacian_tensor


def load_and_preprocess(
    features_path: str | Path,
    labels_path: str | Path,
    relations_path: str | Path,
) -> Tuple[Array2D, Array2D, torch.Tensor, int, int]:
    """Load features, labels, and topology relations."""
    features_file = require_file(features_path, "Feature file")
    labels_file = require_file(labels_path, "Label file")
    relations_file = require_file(relations_path, "Relation file")

    print("Loading data...")

    features_df = load_csv_with_index(features_file, "feature file")
    labels_df = load_csv_with_index(labels_file, "label file")
    labels_df = align_labels_to_features(features_df, labels_df)

    x_raw = features_df.to_numpy(dtype=np.float32)
    if not np.isfinite(x_raw).all():
        raise ValueError("The feature matrix contains NaN or infinite values.")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_raw).astype(np.float32)

    y_encoded, node_to_flat_idx, num_samples, num_branches, max_classes = encode_branch_labels(labels_df)
    laplacian_tensor = build_laplacian(relations_file, node_to_flat_idx, num_branches, max_classes)

    print(
        "Dataset summary: "
        f"samples={num_samples}, features={x_scaled.shape[1]}, "
        f"branches={num_branches}, max_classes={max_classes}"
    )
    return x_scaled, y_encoded, laplacian_tensor, num_branches, max_classes


class PangenomeDataset(Dataset):
    """Dataset wrapper for k-mer features and encoded branch labels."""

    def __init__(self, features: Array2D, labels: Array2D) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


class FastPangenomeNet(nn.Module):
    """Multi-task neural network for branch-node prediction."""

    def __init__(self, input_dim: int, num_branches: int, max_classes: int, hidden_dim: int = 1024) -> None:
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
        self.multi_task_head = nn.Linear(hidden_dim // 2, num_branches * max_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extractor(x)
        logits_flat = self.multi_task_head(features)
        return logits_flat.view(-1, self.num_branches, self.max_classes)


class TopologyAwareLoss(nn.Module):
    """Cross-entropy loss with topology-aware graph smoothness regularization."""

    def __init__(self, laplacian_matrix: torch.Tensor, device: torch.device, lambda_graph: float = 1e-4) -> None:
        super().__init__()
        self.laplacian = laplacian_matrix.to(device)
        self.lambda_graph = lambda_graph
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits_for_ce = logits.transpose(1, 2)
        loss_ce = self.cross_entropy(logits_for_ce, targets)

        probabilities = F.softmax(logits, dim=2)
        probabilities_flat = probabilities.reshape(logits.size(0), -1)
        graph_term = torch.sparse.mm(self.laplacian, probabilities_flat.t()).t()
        loss_graph = torch.mean(torch.sum(probabilities_flat * graph_term, dim=1))

        loss_total = loss_ce + self.lambda_graph * loss_graph
        return loss_total, loss_ce, loss_graph


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: TopologyAwareLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    lambda_graph: float,
) -> dict[str, float]:
    """Run one training epoch and return averaged training metrics."""
    model.train()

    total_loss_sum = 0.0
    ce_loss_sum = 0.0
    graph_loss_sum = 0.0
    weighted_graph_loss_sum = 0.0
    correct = 0
    valid_labels = 0
    num_batches = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss_total, loss_ce, loss_graph = criterion(logits, batch_y)
        loss_total.backward()
        optimizer.step()

        total_loss_sum += loss_total.item()
        ce_loss_sum += loss_ce.item()
        graph_loss_sum += loss_graph.item()
        weighted_graph_loss_sum += lambda_graph * loss_graph.item()
        num_batches += 1

        predictions = torch.argmax(logits, dim=2)
        valid_mask = batch_y != -100
        correct += (predictions[valid_mask] == batch_y[valid_mask]).sum().item()
        valid_labels += valid_mask.sum().item()

    if num_batches == 0:
        raise RuntimeError("The training loader is empty. Check the input files and batch size.")

    return {
        "total_loss": total_loss_sum / num_batches,
        "ce_loss": ce_loss_sum / num_batches,
        "graph_loss": graph_loss_sum / num_batches,
        "weighted_graph_loss": weighted_graph_loss_sum / num_batches,
        "accuracy": correct / valid_labels if valid_labels > 0 else 0.0,
    }


def build_default_model_path() -> Path:
    """Return the default model path in the current working directory."""
    time_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path.cwd() / f"pgpath_model_{time_string}.pth"


def save_model(model: nn.Module, output_path: str | Path | None = None) -> Path:
    """Save the model state dictionary to a user-specified or default path."""
    model_path = build_default_model_path() if output_path is None else Path(output_path)

    if model_path.suffix == "":
        model_path = model_path.with_suffix(".pth")

    if model_path.parent != Path(""):
        model_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_path)
    return model_path


def train_model(args: argparse.Namespace) -> Path:
    """Load data, train the model, and save the trained weights."""
    set_seed(args.seed)
    device = resolve_device(args.device)

    print(f"Device: {device}")
    print(f"Features: {args.features}")
    print(f"Labels: {args.labels}")
    print(f"Relations: {args.relations}")

    x, y, laplacian_tensor, num_branches, max_classes = load_and_preprocess(
        args.features,
        args.labels,
        args.relations,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.seed,
    )
    del x_test, y_test

    train_dataset = PangenomeDataset(x_train, y_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )

    model = FastPangenomeNet(
        input_dim=x.shape[1],
        num_branches=num_branches,
        max_classes=max_classes,
        hidden_dim=args.hidden_dim,
    ).to(device)

    criterion = TopologyAwareLoss(
        laplacian_matrix=laplacian_tensor,
        device=device,
        lambda_graph=args.lambda_graph,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
    )

    print(
        "Training configuration: "
        f"train_samples={len(train_dataset)}, epochs={args.epochs}, "
        f"batch_size={args.batch_size}, lambda_graph={args.lambda_graph}"
    )

    global_start_time = time.time()

    for epoch in range(args.epochs):
        metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            lambda_graph=args.lambda_graph,
        )
        scheduler.step(metrics["total_loss"])

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] | "
            f"CE Loss: {metrics['ce_loss']:.6f} | "
            f"Graph Loss: {metrics['graph_loss']:.6f} | "
            f"Lambda*Graph: {metrics['weighted_graph_loss']:.6f} | "
            f"Total Loss: {metrics['total_loss']:.6f}"
        )

    total_training_time = time.time() - global_start_time
    model_path = save_model(model, args.output)

    print(f"Training finished in {total_training_time / 60:.2f} minutes.")
    print(f"Model saved to: {model_path}")
    return model_path


def main() -> None:
    """Entry point."""
    args = parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
