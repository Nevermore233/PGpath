#!/usr/bin/env python3
"""Build a PGpath population-level k-mer feature vector from paired-end FASTQ files.

For each paired-end sample, this script counts k-mers with Jellyfish, queries the
selected PGpath k-mers, normalizes counts within that sample, and averages the
normalized vectors across samples. The output is a one-row CSV suitable as input
for PGpath inference.

Jellyfish databases are created in a run-specific temporary directory. Unless
--keep-temp is enabled, each sample database is removed immediately after its
selected k-mers have been queried. This limits temporary disk usage to roughly
one sample database at a time.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


FASTQ_SUFFIXES = (
    ".fastq.gz",
    ".fq.gz",
    ".fastq",
    ".fq",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Count selected PGpath k-mers from paired-end FASTQ files and build "
            "a population-level normalized k-mer frequency CSV."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        required=True,
        help="Directory containing paired-end FASTQ files.",
    )
    parser.add_argument(
        "-k",
        "--kmers",
        required=True,
        help="Text file containing selected k-mers, one k-mer per line.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="new_population.csv",
        help="Output one-row population feature CSV.",
    )
    parser.add_argument(
        "-n",
        "--population-name",
        default="new_population",
        help="Row name used in the output CSV.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search FASTQ files recursively under the input directory.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=16,
        help="Number of Jellyfish threads.",
    )
    parser.add_argument(
        "--hash-size",
        default="1G",
        help="Jellyfish hash size, for example 500M, 1G, or 4G.",
    )
    parser.add_argument(
        "--jellyfish",
        default="jellyfish",
        help="Path to the Jellyfish executable.",
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
        help=(
            "Parent directory for run-specific Jellyfish temporary databases. "
            "If omitted, the operating system temporary directory is used."
        ),
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help=(
            "Keep the run-specific Jellyfish temporary directory and all sample "
            "databases for debugging."
        ),
    )
    parser.add_argument(
        "--query-chunk-size",
        type=int,
        default=500,
        help="Number of k-mers queried per Jellyfish command.",
    )
    parser.add_argument(
        "--save-sample-matrix",
        default=None,
        help="Optional output CSV for per-sample normalized k-mer frequencies.",
    )
    parser.add_argument(
        "--skip-zero-count-samples",
        action="store_true",
        help=(
            "Skip samples whose selected k-mer count sum is zero instead of "
            "raising an error."
        ),
    )
    return parser.parse_args()


def require_file(path: str, label: str) -> Path:
    """Resolve and validate a required input file."""
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {file_path}")
    return file_path


def require_dir(path: str, label: str) -> Path:
    """Resolve and validate a required input directory."""
    dir_path = Path(path).expanduser().resolve()
    if not dir_path.is_dir():
        raise NotADirectoryError(f"{label} does not exist or is not a directory: {dir_path}")
    return dir_path


def resolve_output_path(path: str) -> Path:
    """Resolve an output path and create its parent directory when necessary."""
    output_path = Path(path).expanduser().resolve()
    if output_path.suffix == "":
        output_path = output_path.with_suffix(".csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def remove_fastq_suffix(path: Path) -> str:
    """Remove a supported FASTQ suffix from a file name."""
    name = path.name
    for suffix in FASTQ_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def is_fastq_file(path: Path) -> bool:
    """Return whether a path is a supported FASTQ file."""
    return path.is_file() and any(path.name.endswith(suffix) for suffix in FASTQ_SUFFIXES)


def infer_sample_and_mate(path: Path) -> Tuple[str, str] | None:
    """Infer the sample name and mate identifier from a FASTQ file name."""
    base = remove_fastq_suffix(path)
    patterns = [
        r"^(?P<sample>.+?)(?:[._-]L\d{3})?[._-]R(?P<mate>[12])(?:[._-]\d{3})?$",
        r"^(?P<sample>.+?)(?:[._-]L\d{3})?[._-](?P<mate>[12])(?:[._-]\d{3})?$",
    ]
    for pattern in patterns:
        match = re.match(pattern, base, flags=re.IGNORECASE)
        if match:
            return match.group("sample"), match.group("mate")
    return None


def discover_fastq_pairs(
    input_dir: Path,
    recursive: bool,
) -> List[Tuple[str, List[Path], List[Path]]]:
    """Discover paired-end FASTQ samples and group lane-split files by sample."""
    paths = input_dir.rglob("*") if recursive else input_dir.iterdir()
    grouped: DefaultDict[str, DefaultDict[str, List[Path]]] = defaultdict(
        lambda: defaultdict(list)
    )
    ignored = 0

    for path in paths:
        if not is_fastq_file(path):
            continue
        parsed = infer_sample_and_mate(path)
        if parsed is None:
            ignored += 1
            continue
        sample_name, mate = parsed
        grouped[sample_name][mate].append(path.resolve())

    pairs: List[Tuple[str, List[Path], List[Path]]] = []
    incomplete_samples: List[str] = []

    for sample_name in sorted(grouped):
        r1_files = sorted(grouped[sample_name].get("1", []))
        r2_files = sorted(grouped[sample_name].get("2", []))
        if r1_files and r2_files:
            pairs.append((sample_name, r1_files, r2_files))
        else:
            incomplete_samples.append(sample_name)

    if not pairs:
        raise ValueError(
            "No valid paired-end FASTQ samples were found. Expected file names such as "
            "sample_R1.fastq.gz and sample_R2.fastq.gz, or sample_1.fq and sample_2.fq."
        )

    if ignored > 0:
        print(f"Ignored FASTQ files with unrecognized mate naming: {ignored}")
    if incomplete_samples:
        preview = ", ".join(incomplete_samples[:10])
        suffix = "..." if len(incomplete_samples) > 10 else ""
        print(
            "Ignored samples missing R1 or R2 files: "
            f"{len(incomplete_samples)} ({preview}{suffix})"
        )

    return pairs


def read_kmers(path: Path) -> List[str]:
    """Read and validate the selected k-mer list."""
    kmers: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            kmer = line.strip()
            if not kmer or kmer.startswith("#"):
                continue
            if re.search(r"\s", kmer):
                raise ValueError(f"Invalid whitespace in k-mer at {path}:{line_number}")
            kmers.append(kmer.upper())

    if not kmers:
        raise ValueError(f"No k-mers were found in: {path}")
    if len(set(kmers)) != len(kmers):
        raise ValueError("Duplicated k-mers were found in the selected k-mer file.")

    lengths = {len(kmer) for kmer in kmers}
    if len(lengths) != 1:
        raise ValueError(
            "All selected k-mers must have the same length. "
            f"Found lengths: {sorted(lengths)}"
        )

    for kmer in kmers:
        if not re.fullmatch(r"[ACGTN]+", kmer):
            raise ValueError(f"Invalid k-mer sequence: {kmer}")

    return kmers


def ensure_jellyfish(executable: str) -> str:
    """Resolve the Jellyfish executable."""
    path = shutil.which(executable)
    if path is None:
        raise FileNotFoundError(
            f"Jellyfish executable was not found: {executable}. "
            "Install Jellyfish or provide its path with --jellyfish."
        )
    return path


def run_command(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run an external command and report its stderr on failure."""
    try:
        return subprocess.run(
            list(command),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip()
        stdout = exc.stdout.strip()
        detail = stderr if stderr else stdout
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{detail}") from exc


def chunked(items: Sequence[str], chunk_size: int) -> Iterable[Sequence[str]]:
    """Yield fixed-size chunks from a sequence."""
    if chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive.")
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def sanitize_sample_name(sample_name: str) -> str:
    """Create a filesystem-safe sample identifier for temporary files."""
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", sample_name)
    return safe_name or "sample"


def remove_sample_jellyfish_files(temp_dir: Path, sample_name: str) -> None:
    """Remove Jellyfish files generated for one sample."""
    for database_file in temp_dir.glob(f"{sample_name}.jf*"):
        try:
            database_file.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(
                f"Warning: failed to remove temporary Jellyfish file "
                f"{database_file}: {exc}",
                file=sys.stderr,
            )


def count_sample_kmers(
    jellyfish: str,
    kmers: List[str],
    sample_name: str,
    read_files: List[Path],
    temp_dir: Path,
    threads: int,
    hash_size: str,
    query_chunk_size: int,
) -> np.ndarray:
    """Count all k-mers for one sample and query the selected PGpath k-mers."""
    k = len(kmers[0])
    jf_path = temp_dir / f"{sample_name}.jf"

    if jf_path.exists():
        jf_path.unlink()

    count_command = [
        jellyfish,
        "count",
        "-m",
        str(k),
        "-s",
        str(hash_size),
        "-t",
        str(threads),
        "-C",
        "-o",
        str(jf_path),
    ] + [str(path) for path in read_files]
    run_command(count_command)

    if not jf_path.is_file() or jf_path.stat().st_size == 0:
        raise RuntimeError(
            f"Jellyfish did not produce a valid database for sample '{sample_name}': {jf_path}"
        )

    counts = {kmer: 0 for kmer in kmers}
    for kmer_chunk in chunked(kmers, query_chunk_size):
        query_command = [jellyfish, "query", str(jf_path)] + list(kmer_chunk)
        result = run_command(query_command)
        for line in result.stdout.splitlines():
            fields = line.strip().split()
            if len(fields) < 2:
                continue
            kmer = fields[0].upper()
            try:
                count = int(fields[1])
            except ValueError:
                continue
            if kmer in counts:
                counts[kmer] = count

    return np.array([counts[kmer] for kmer in kmers], dtype=np.float64)


def create_run_temp_dir(parent_temp: Path | None) -> Path:
    """Create a unique temporary directory for the current PGpath run."""
    if parent_temp is not None:
        parent_temp.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="pgpath_jellyfish_", dir=parent_temp))


def build_population_features(args: argparse.Namespace) -> Path:
    """Build and save a population-level normalized k-mer feature vector."""
    input_dir = require_dir(args.input_dir, "Input FASTQ directory")
    kmer_path = require_file(args.kmers, "Selected k-mer file")
    output_path = resolve_output_path(args.output)
    sample_matrix_path = (
        resolve_output_path(args.save_sample_matrix)
        if args.save_sample_matrix
        else None
    )

    if args.threads <= 0:
        raise ValueError("threads must be positive.")
    if args.query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive.")

    jellyfish = ensure_jellyfish(args.jellyfish)
    kmers = read_kmers(kmer_path)
    pairs = discover_fastq_pairs(input_dir, args.recursive)

    print(f"Selected k-mers: {len(kmers):,} (k={len(kmers[0])})")
    print(f"Detected paired-end samples: {len(pairs):,}")

    parent_temp = (
        Path(args.temp_dir).expanduser().resolve()
        if args.temp_dir
        else None
    )
    temp_dir = create_run_temp_dir(parent_temp)
    print(f"Jellyfish temporary directory: {temp_dir}")

    sample_vectors: List[np.ndarray] = []
    used_sample_names: List[str] = []

    try:
        for sample_index, (sample_name, r1_files, r2_files) in enumerate(pairs, start=1):
            read_files = r1_files + r2_files
            safe_sample_name = f"{sample_index:04d}_{sanitize_sample_name(sample_name)}"

            print(
                f"Counting sample: {sample_name} "
                f"({len(read_files)} FASTQ file(s))"
            )

            try:
                counts = count_sample_kmers(
                    jellyfish=jellyfish,
                    kmers=kmers,
                    sample_name=safe_sample_name,
                    read_files=read_files,
                    temp_dir=temp_dir,
                    threads=args.threads,
                    hash_size=args.hash_size,
                    query_chunk_size=args.query_chunk_size,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to count or query selected k-mers for sample "
                    f"'{sample_name}'. Temporary database directory: {temp_dir}. "
                    "Check available disk space and filesystem quotas, or rerun with "
                    "--temp-dir pointing to a filesystem with sufficient free space."
                ) from exc
            finally:
                if not args.keep_temp:
                    remove_sample_jellyfish_files(temp_dir, safe_sample_name)

            total = float(counts.sum())
            if total <= 0:
                message = f"Selected k-mer count sum is zero for sample: {sample_name}"
                if args.skip_zero_count_samples:
                    print(f"Warning: {message}; skipped.")
                    continue
                raise ValueError(message)

            sample_vectors.append(counts / total)
            used_sample_names.append(sample_name)

        if not sample_vectors:
            raise ValueError("No valid sample-level k-mer vectors were generated.")

        sample_matrix = np.vstack(sample_vectors)
        population_vector = sample_matrix.mean(axis=0)

        output_df = pd.DataFrame(
            [population_vector],
            index=[args.population_name],
            columns=kmers,
        )
        output_df.to_csv(output_path)

        if sample_matrix_path is not None:
            sample_df = pd.DataFrame(
                sample_matrix,
                index=used_sample_names,
                columns=kmers,
            )
            sample_df.to_csv(sample_matrix_path)

        print(f"Used samples: {len(used_sample_names):,}")
        print(f"Saved population feature CSV: {output_path}")
        if sample_matrix_path is not None:
            print(f"Saved per-sample feature CSV: {sample_matrix_path}")

    finally:
        if args.keep_temp:
            print(f"Temporary files kept at: {temp_dir}")
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)

    return output_path


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    try:
        build_population_features(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        cause = exc.__cause__
        if cause is not None:
            print(f"Caused by: {cause}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
