#!/usr/bin/env python3
"""Run the PGpath post-training pipeline in one command.

This wrapper performs three steps:
1. Build a population-level k-mer feature vector from paired-end FASTQ files.
2. Run PGpath branch-node inference with a trained model.
3. Reconstruct a PGpath-derived linear reference FASTA from the predicted branch nodes.

The script expects the helper modules and default PGpath resources to be
available in the same directory as this file, unless overridden by command-line
arguments.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import sys
from argparse import Namespace
from pathlib import Path

from pgpath_kmer_profile import build_population_features
from pgpath_infer import run_inference
from pgpath_reconstruct import (
    DEFAULT_FASTA_LINE_WIDTH,
    DEFAULT_MAX_ALT_STEPS,
    DEFAULT_MAX_SKIP_BP,
    reconstruct_anchored_genome,
)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_KMERS = SCRIPT_DIR / "pgpath_selected_kmers.txt"
DEFAULT_SCALER_STATS = SCRIPT_DIR / "pgpath_scaler_stats.csv"
DEFAULT_MODEL = SCRIPT_DIR / "trained_model.pth"
DEFAULT_LABELS = SCRIPT_DIR / "labels.csv"
DEFAULT_GFA = SCRIPT_DIR / "pangenome_graph_default.gfa"


def resolve_path(path: str | None) -> Path | None:
    """Resolve a user-provided path while preserving None."""
    if path is None:
        return None
    return Path(path).expanduser().resolve()


def require_existing_file(path: str, label: str) -> str:
    """Return an existing file path as a string for downstream modules."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"{label} does not exist or is not a file: {resolved}"
        )
    return str(resolved)


def validate_resource_files(args: argparse.Namespace) -> None:
    """Validate bundled or user-provided PGpath resource files."""
    args.kmers = require_existing_file(args.kmers, "Selected k-mer file")
    args.scaler_stats = require_existing_file(args.scaler_stats, "Scaler statistics file")
    args.model = require_existing_file(args.model, "Model file")
    args.labels = require_existing_file(args.labels, "Label file")
    args.gfa = require_existing_file(args.gfa, "GFA file")


def make_default_work_dir() -> Path:
    """Create a timestamped working directory for intermediate files."""
    timestamp = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return (Path.cwd() / f"pgpath_run_{timestamp}").resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the integrated PGpath pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Run PGpath from paired-end FASTQ files to a reconstructed "
            "population-adapted linear reference FASTA."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    required = parser.add_argument_group("Required inputs")
    required.add_argument(
        "-i",
        "--input-dir",
        required=True,
        help="Directory containing paired-end FASTQ files for the target population.",
    )
    required.add_argument(
        "-o",
        "--output",
        required=True,
        help="Final output FASTA path.",
    )

    resources = parser.add_argument_group("PGpath resource files")
    resources.add_argument(
        "-k",
        "--kmers",
        default=str(DEFAULT_KMERS),
        help="Selected PGpath k-mer list, one k-mer per line.",
    )
    resources.add_argument(
        "--scaler-stats",
        default=str(DEFAULT_SCALER_STATS),
        help="Scaler statistics CSV exported from the training feature matrix.",
    )
    resources.add_argument(
        "-m",
        "--model",
        default=str(DEFAULT_MODEL),
        help="Trained PGpath model weights (.pth).",
    )
    resources.add_argument(
        "-l",
        "--labels",
        default=str(DEFAULT_LABELS),
        help="Original branch label CSV used to rebuild label mappings.",
    )
    resources.add_argument(
        "-g",
        "--gfa",
        default=str(DEFAULT_GFA),
        help="Input pangenome graph in GFA format.",
    )

    output = parser.add_argument_group("Run control")
    output.add_argument(
        "--work-dir",
        default=None,
        help="Directory for intermediate feature and prediction CSV files.",
    )
    output.add_argument(
        "-n",
        "--population-name",
        default="new_population",
        help="Population name used as the row index in intermediate CSV files.",
    )
    output.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep intermediate feature and prediction CSV files after completion.",
    )
    output.add_argument(
        "--population-feature-output",
        default=None,
        help="Optional path for the intermediate population k-mer feature CSV.",
    )
    output.add_argument(
        "--prediction-output",
        default=None,
        help="Optional path for the intermediate predicted branch-node CSV.",
    )

    kmer_count = parser.add_argument_group("K-mer counting options")
    kmer_count.add_argument(
        "--recursive",
        action="store_true",
        help="Search FASTQ files recursively under the input directory.",
    )
    kmer_count.add_argument(
        "--threads",
        type=int,
        default=16,
        help="Number of Jellyfish threads.",
    )
    kmer_count.add_argument(
        "--hash-size",
        default="1G",
        help="Jellyfish hash size, for example 500M, 1G, or 4G.",
    )
    kmer_count.add_argument(
        "--jellyfish",
        default="jellyfish",
        help="Path to the Jellyfish executable.",
    )
    kmer_count.add_argument(
        "--temp-dir",
        default=None,
        help="Temporary directory for Jellyfish databases.",
    )
    kmer_count.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary Jellyfish databases for debugging.",
    )
    kmer_count.add_argument(
        "--query-chunk-size",
        type=int,
        default=500,
        help="Number of k-mers queried per Jellyfish command.",
    )
    kmer_count.add_argument(
        "--save-sample-matrix",
        default=None,
        help="Optional output CSV for per-sample normalized k-mer frequencies.",
    )
    kmer_count.add_argument(
        "--skip-zero-count-samples",
        action="store_true",
        help="Skip samples with zero selected-k-mer counts instead of raising an error.",
    )

    inference = parser.add_argument_group("Inference options")
    inference.add_argument(
        "--hidden-dim",
        type=int,
        default=1024,
        help="Hidden dimension of the model. Must match the training setting.",
    )
    inference.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Inference batch size.",
    )
    inference.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device for inference.",
    )

    reconstruction = parser.add_argument_group("Reference reconstruction options")
    reconstruction.add_argument(
        "--max-skip-bp",
        type=int,
        default=DEFAULT_MAX_SKIP_BP,
        help="Maximum skipped backbone bases allowed for one branch substitution.",
    )
    reconstruction.add_argument(
        "--max-alt-steps",
        type=int,
        default=DEFAULT_MAX_ALT_STEPS,
        help="Maximum graph steps followed along one alternative path.",
    )
    reconstruction.add_argument(
        "--line-width",
        type=int,
        default=DEFAULT_FASTA_LINE_WIDTH,
        help="FASTA line width.",
    )
    reconstruction.add_argument(
        "--chrom-name-style",
        choices=("as-is", "chm13"),
        default="chm13",
        help=(
            "Chromosome names used in the output FASTA. Use 'chm13' to convert "
            "known T2T-CHM13 RefSeq accessions such as NC_060925.1 to chr1."
        ),
    )
    reconstruction.add_argument(
        "--chrom-map",
        default=None,
        help=(
            "Optional CSV/TSV file mapping GFA chromosome names to FASTA header names. "
            "Use columns 'source' and 'target', or use the first two columns."
        ),
    )
    reconstruction.add_argument(
        "--allow-unmapped-chroms",
        action="store_true",
        help="Keep chromosome names unchanged if not found in the selected mapping.",
    )

    return parser


def run_pipeline(args: argparse.Namespace) -> Path:
    """Run population feature construction, inference, and reconstruction."""
    validate_resource_files(args)

    work_dir = resolve_path(args.work_dir) or make_default_work_dir()
    work_dir.mkdir(parents=True, exist_ok=True)

    feature_output = resolve_path(args.population_feature_output)
    if feature_output is None:
        feature_output = work_dir / f"{args.population_name}.features.csv"

    prediction_output = resolve_path(args.prediction_output)
    if prediction_output is None:
        prediction_output = work_dir / f"{args.population_name}.predicted_paths.csv"

    fasta_output = resolve_path(args.output)
    if fasta_output is None:
        timestamp = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fasta_output = Path.cwd() / f"PGpath_reference_{timestamp}.fasta"

    print("Step 1/3: Building population-level k-mer features.")
    build_args = Namespace(
        input_dir=args.input_dir,
        kmers=args.kmers,
        output=str(feature_output),
        population_name=args.population_name,
        recursive=args.recursive,
        threads=args.threads,
        hash_size=args.hash_size,
        jellyfish=args.jellyfish,
        temp_dir=args.temp_dir,
        keep_temp=args.keep_temp,
        query_chunk_size=args.query_chunk_size,
        save_sample_matrix=args.save_sample_matrix,
        skip_zero_count_samples=args.skip_zero_count_samples,
    )
    population_feature_csv = build_population_features(build_args)

    print("Step 2/3: Running PGpath branch-node inference.")
    inference_args = Namespace(
        model=args.model,
        input=str(population_feature_csv),
        training_resource=args.kmers,
        scaler_stats=args.scaler_stats,
        labels=args.labels,
        output=str(prediction_output),
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        device=args.device,
    )
    prediction_csv = run_inference(inference_args)

    print("Step 3/3: Reconstructing PGpath-derived reference FASTA.")
    reconstruct_anchored_genome(
        gfa_path=args.gfa,
        pred_csv_path=str(prediction_csv),
        target_sample=args.population_name,
        output_fasta=str(fasta_output),
        max_skip_bp=args.max_skip_bp,
        max_alt_steps=args.max_alt_steps,
        line_width=args.line_width,
        chrom_name_style=args.chrom_name_style,
        chrom_map_path=args.chrom_map,
        allow_unmapped_chroms=args.allow_unmapped_chroms,
    )

    if not args.keep_intermediate:
        for path in (feature_output, prediction_output):
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass
        try:
            if work_dir.exists() and not any(work_dir.iterdir()):
                work_dir.rmdir()
        except OSError:
            pass

    print(f"PGpath pipeline completed. Final FASTA: {fasta_output}")
    return fasta_output


def main() -> None:
    args = build_arg_parser().parse_args()
    try:
        run_pipeline(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
