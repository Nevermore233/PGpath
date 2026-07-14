#!/usr/bin/env python3
"""Reconstruct a PGpath-derived linear reference from predicted branch nodes.

The script traverses primary backbone nodes in a GFA graph and inserts predicted
non-primary branch paths only when they connect from the current backbone node
and rejoin a downstream backbone node on the same chromosome.
"""

import argparse
import datetime
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd


DEFAULT_MAX_SKIP_BP = 500_000
DEFAULT_MAX_ALT_STEPS = 1_000
DEFAULT_FASTA_LINE_WIDTH = 80

# T2T-CHM13 v2.0 RefSeq chromosome accessions. These mappings are used
# only when --chrom-name-style chm13 is selected.
CHM13_REFSEQ_CHROM_MAP = {
    "NC_060925.1": "chr1",
    "NC_060926.1": "chr2",
    "NC_060927.1": "chr3",
    "NC_060928.1": "chr4",
    "NC_060929.1": "chr5",
    "NC_060930.1": "chr6",
    "NC_060931.1": "chr7",
    "NC_060932.1": "chr8",
    "NC_060933.1": "chr9",
    "NC_060934.1": "chr10",
    "NC_060935.1": "chr11",
    "NC_060936.1": "chr12",
    "NC_060937.1": "chr13",
    "NC_060938.1": "chr14",
    "NC_060939.1": "chr15",
    "NC_060940.1": "chr16",
    "NC_060941.1": "chr17",
    "NC_060942.1": "chr18",
    "NC_060943.1": "chr19",
    "NC_060944.1": "chr20",
    "NC_060945.1": "chr21",
    "NC_060946.1": "chr22",
    "NC_060947.1": "chrX",
    "NC_060948.1": "chrY",
}


def extract_tag(parts: Sequence[str], prefix: str) -> Optional[str]:
    """Extract the value of a GFA optional tag."""
    for part in parts:
        if part.startswith(prefix):
            fields = part.split(":", 2)
            if len(fields) == 3:
                return fields[2]
    return None


def natural_sort_key(value: str) -> List[object]:
    """Return a natural-sort key, e.g. chr2 before chr10."""
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"([0-9]+)", value)
    ]


def validate_input_file(path: str, description: str) -> Path:
    """Validate that an input file exists and is readable."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"{description} does not exist or is not a file: {path}")
    if not os.access(file_path, os.R_OK):
        raise PermissionError(f"{description} is not readable: {path}")
    return file_path


def load_chromosome_name_map(
    chrom_name_style: str = "as-is",
    chrom_map_path: Optional[str] = None,
) -> Dict[str, str]:
    """Load chromosome-name mappings for FASTA headers.

    The mapping changes only FASTA record names. Internal reconstruction still
    uses the chromosome identifiers present in the GFA file.
    """
    if chrom_name_style not in {"as-is", "chm13"}:
        raise ValueError("chrom_name_style must be either 'as-is' or 'chm13'.")

    chrom_map: Dict[str, str] = {}
    if chrom_name_style == "chm13":
        chrom_map.update(CHM13_REFSEQ_CHROM_MAP)

    if chrom_map_path is not None:
        map_file = validate_input_file(chrom_map_path, "Chromosome-name map file")
        try:
            map_df = pd.read_csv(map_file, sep=None, engine="python")
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"Chromosome-name map file is empty: {map_file}") from exc

        if map_df.empty or map_df.shape[1] < 2:
            raise ValueError(
                "Chromosome-name map file must contain at least two columns. "
                "Use columns named 'source' and 'target', or provide the source "
                "and target names as the first two columns."
            )

        if {"source", "target"}.issubset(map_df.columns):
            source_col, target_col = "source", "target"
        else:
            source_col, target_col = map_df.columns[:2]

        for _, row in map_df.iterrows():
            source = str(row[source_col]).strip()
            target = str(row[target_col]).strip()
            if source and target and source.lower() != "nan" and target.lower() != "nan":
                chrom_map[source] = target

    return chrom_map


def build_output_chromosome_names(
    chroms: Iterable[str],
    chrom_map: Dict[str, str],
    allow_unmapped_chroms: bool = False,
) -> Dict[str, str]:
    """Resolve output chromosome names and prevent ambiguous FASTA headers."""
    output_names: Dict[str, str] = {}
    reverse_names: Dict[str, str] = {}
    unmapped: List[str] = []

    for chrom in chroms:
        output_name = chrom_map.get(chrom, chrom)
        if chrom_map and chrom not in chrom_map:
            unmapped.append(chrom)

        if output_name in reverse_names and reverse_names[output_name] != chrom:
            raise ValueError(
                "Chromosome-name mapping is ambiguous: "
                f"'{reverse_names[output_name]}' and '{chrom}' both map to '{output_name}'."
            )
        reverse_names[output_name] = chrom
        output_names[chrom] = output_name

    if unmapped and not allow_unmapped_chroms:
        preview = ", ".join(sorted(unmapped, key=natural_sort_key)[:10])
        raise ValueError(
            "Some GFA chromosome names were not found in the selected mapping: "
            f"{preview}. Use --allow-unmapped-chroms to keep unmatched names unchanged, "
            "or provide a complete mapping with --chrom-map."
        )

    return output_names


def resolve_output_path(output_path: Optional[str]) -> Path:
    """Create a timestamped FASTA path if no output path is provided."""
    if output_path is None:
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path.cwd() / f"PGpath_reference_{time_str}.fasta"

    output = Path(output_path)
    if output.exists() and output.is_dir():
        raise IsADirectoryError(f"Output path points to a directory: {output_path}")
    if output.parent != Path(""):
        output.parent.mkdir(parents=True, exist_ok=True)
    return output


def load_predicted_nodes(
    pred_csv_path: Path,
    target_sample: Optional[str] = None,
) -> Tuple[Set[str], str]:
    """Load predicted branch-node labels and resolve the target sample name."""
    try:
        preds_df = pd.read_csv(pred_csv_path, index_col=0)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Prediction CSV is empty: {pred_csv_path}") from exc

    if preds_df.empty:
        raise ValueError(f"Prediction CSV contains no records: {pred_csv_path}")

    if target_sample is None:
        if len(preds_df.index) != 1:
            preview = ", ".join(map(str, preds_df.index[:5]))
            raise ValueError(
                "--sample is required when the prediction CSV contains multiple rows. "
                f"Available sample examples: {preview}"
            )
        target_sample = str(preds_df.index[0])

    if target_sample not in preds_df.index:
        preview = ", ".join(map(str, preds_df.index[:5]))
        raise ValueError(
            f"Target sample '{target_sample}' was not found in the prediction CSV. "
            f"Available sample examples: {preview}"
        )

    sample_row = preds_df.loc[target_sample]
    if isinstance(sample_row, pd.DataFrame):
        raise ValueError(
            f"Target sample '{target_sample}' appears multiple times in the prediction CSV index."
        )

    predicted_nodes = {
        str(value)
        for value in sample_row.dropna().values
        if str(value).strip() != ""
    }
    if not predicted_nodes:
        raise ValueError(f"No predicted nodes were found for target sample '{target_sample}'.")

    return predicted_nodes, target_sample


def parse_gfa(
    gfa_path: Path,
) -> Tuple[
    Dict[str, str],
    Dict[str, List[str]],
    Set[str],
    DefaultDict[str, List[str]],
    Dict[str, Tuple[str, int]],
]:
    """Parse GFA segments, primary backbone nodes, and directed links."""
    segments_seq: Dict[str, str] = {}
    backbone_nodes: DefaultDict[str, List[Tuple[int, str]]] = defaultdict(list)
    is_primary: Set[str] = set()
    adj_list: DefaultDict[str, List[str]] = defaultdict(list)

    segment_count = 0
    link_count = 0

    with gfa_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line or line.startswith("#"):
                continue

            if line.startswith("S\t"):
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    raise ValueError(f"Malformed segment line at {gfa_path}:{line_number}")

                node_id = parts[1]
                seq = parts[2]
                chrom = extract_tag(parts, "SN:Z:")
                offset_str = extract_tag(parts, "SO:i:")
                sr = extract_tag(parts, "SR:i:")

                if seq != "*":
                    segments_seq[node_id] = seq

                if sr == "0":
                    is_primary.add(node_id)
                    if chrom and offset_str is not None:
                        try:
                            offset = int(offset_str)
                        except ValueError as exc:
                            raise ValueError(
                                f"Invalid SO:i offset at {gfa_path}:{line_number}: {offset_str}"
                            ) from exc
                        backbone_nodes[chrom].append((offset, node_id))

                segment_count += 1

            elif line.startswith("L\t"):
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    raise ValueError(f"Malformed link line at {gfa_path}:{line_number}")

                src = parts[1]
                tgt = parts[3]
                adj_list[src].append(tgt)
                link_count += 1

    if segment_count == 0:
        raise ValueError(f"No segment records were found in the GFA file: {gfa_path}")
    if not backbone_nodes:
        raise ValueError(
            "No primary backbone nodes with SR:i:0 and coordinate tags were found in the GFA file."
        )

    ordered_backbone: Dict[str, List[str]] = {}
    node_to_backbone_pos: Dict[str, Tuple[str, int]] = {}

    for chrom, nodes in backbone_nodes.items():
        nodes.sort(key=lambda item: item[0])
        node_list = [node_id for _, node_id in nodes]
        ordered_backbone[chrom] = node_list
        for idx, node_id in enumerate(node_list):
            if node_id in node_to_backbone_pos:
                raise ValueError(f"Backbone node appears more than once: {node_id}")
            node_to_backbone_pos[node_id] = (chrom, idx)

    print(
        "Parsed GFA: "
        f"{segment_count:,} segments, {link_count:,} links, "
        f"{len(ordered_backbone):,} chromosomes."
    )

    return segments_seq, ordered_backbone, is_primary, adj_list, node_to_backbone_pos


def write_fasta_record(
    handle,
    chrom: str,
    sequence_pieces: Iterable[str],
    line_width: int,
) -> int:
    """Write one FASTA record and return its sequence length."""
    if line_width <= 0:
        raise ValueError("FASTA line width must be positive.")

    handle.write(f">{chrom}\n")
    full_sequence = "".join(sequence_pieces)
    for start in range(0, len(full_sequence), line_width):
        handle.write(full_sequence[start : start + line_width] + "\n")
    return len(full_sequence)


def reconstruct_anchored_genome(
    gfa_path: str,
    pred_csv_path: str,
    target_sample: Optional[str],
    output_fasta: Optional[str],
    max_skip_bp: int = DEFAULT_MAX_SKIP_BP,
    max_alt_steps: int = DEFAULT_MAX_ALT_STEPS,
    line_width: int = DEFAULT_FASTA_LINE_WIDTH,
    chrom_name_style: str = "as-is",
    chrom_map_path: Optional[str] = None,
    allow_unmapped_chroms: bool = False,
) -> None:
    """Reconstruct a backbone-guided PGpath reference genome."""
    if max_skip_bp < 0:
        raise ValueError("max_skip_bp must be non-negative.")
    if max_alt_steps <= 0:
        raise ValueError("max_alt_steps must be positive.")

    start_time = time.time()

    gfa_file = validate_input_file(gfa_path, "GFA file")
    pred_file = validate_input_file(pred_csv_path, "Prediction CSV")
    output_file = resolve_output_path(output_fasta)

    print(f"GFA file: {gfa_file}")
    print(f"Prediction CSV: {pred_file}")
    print(f"Output FASTA: {output_file}")

    predicted_nodes, resolved_sample = load_predicted_nodes(pred_file, target_sample)
    print(f"Target sample: {resolved_sample}")
    print(f"Loaded predicted nodes: {len(predicted_nodes):,}")

    (
        segments_seq,
        ordered_backbone,
        is_primary,
        adj_list,
        node_to_backbone_pos,
    ) = parse_gfa(gfa_file)

    chrom_name_map = load_chromosome_name_map(
        chrom_name_style=chrom_name_style,
        chrom_map_path=chrom_map_path,
    )
    output_chrom_names = build_output_chromosome_names(
        ordered_backbone.keys(),
        chrom_name_map,
        allow_unmapped_chroms=allow_unmapped_chroms,
    )
    if chrom_name_map:
        converted_count = sum(
            1 for chrom, out_name in output_chrom_names.items() if chrom != out_name
        )
        print(f"Chromosome names converted for FASTA headers: {converted_count:,}")

    total_sv_count = 0
    total_intercept_count = 0

    with output_file.open("w", encoding="utf-8") as out_handle:
        sorted_chroms = sorted(
            ordered_backbone.keys(),
            key=lambda name: natural_sort_key(output_chrom_names[name]),
        )

        for chrom in sorted_chroms:
            chrom_start_time = time.time()
            chrom_seq_pieces: List[str] = []
            bb_list = ordered_backbone[chrom]
            bb_len = len(bb_list)
            i = 0
            sv_count = 0
            intercept_count = 0

            while i < bb_len:
                curr_bb_node = bb_list[i]

                if curr_bb_node in segments_seq:
                    chrom_seq_pieces.append(segments_seq[curr_bb_node])

                next_edges = adj_list.get(curr_bb_node, [])
                chosen_alt = None
                for node_id in next_edges:
                    if node_id not in is_primary and node_id in predicted_nodes:
                        chosen_alt = node_id
                        break

                if chosen_alt is None:
                    i += 1
                    continue

                alt_curr = chosen_alt
                merge_idx = -1
                safety_counter = 0
                temp_alt_seqs: List[str] = []

                while alt_curr and safety_counter < max_alt_steps:
                    safety_counter += 1

                    if alt_curr in segments_seq:
                        temp_alt_seqs.append(segments_seq[alt_curr])

                    next_alt_edges = adj_list.get(alt_curr, [])
                    if not next_alt_edges:
                        break

                    backbone_hits = [node_id for node_id in next_alt_edges if node_id in is_primary]
                    if backbone_hits:
                        hit_node = backbone_hits[0]
                        hit_pos = node_to_backbone_pos.get(hit_node)
                        if hit_pos is not None:
                            hit_chrom, hit_idx = hit_pos
                            if hit_chrom == chrom:
                                merge_idx = hit_idx
                                break

                    next_step = None
                    for node_id in next_alt_edges:
                        if node_id in predicted_nodes:
                            next_step = node_id
                            break
                    if next_step is None:
                        next_step = next_alt_edges[0]
                    alt_curr = next_step

                if merge_idx > i:
                    skipped_bp = sum(
                        len(segments_seq.get(bb_list[k], ""))
                        for k in range(i + 1, merge_idx)
                    )

                    if skipped_bp > max_skip_bp:
                        intercept_count += 1
                        i += 1
                    else:
                        chrom_seq_pieces.extend(temp_alt_seqs)
                        i = merge_idx
                        sv_count += 1
                        continue
                else:
                    i += 1

            output_chrom = output_chrom_names[chrom]
            chrom_length = write_fasta_record(
                out_handle,
                output_chrom,
                chrom_seq_pieces,
                line_width,
            )
            total_sv_count += sv_count
            total_intercept_count += intercept_count
            elapsed_chrom = time.time() - chrom_start_time

            print(
                f"Assembled {chrom} as {output_chrom}: length={chrom_length:,} bp, "
                f"inserted_SV={sv_count:,}, intercepted_jumps={intercept_count:,}, "
                f"time={elapsed_chrom:.2f}s"
            )

    elapsed = time.time() - start_time
    print(
        "Finished reconstruction: "
        f"output={output_file}, total_inserted_SV={total_sv_count:,}, "
        f"total_intercepted_jumps={total_intercept_count:,}, time={elapsed:.2f}s"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Reconstruct a PGpath-derived linear reference from a GFA graph and predicted branch nodes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-g",
        "--gfa",
        required=True,
        help="Input pangenome graph in GFA format.",
    )
    parser.add_argument(
        "-p",
        "--predictions",
        required=True,
        help="CSV file containing predicted branch-node labels. The first column is treated as the sample index.",
    )
    parser.add_argument(
        "-s",
        "--sample",
        default=None,
        help=(
            "Target sample or population name to reconstruct. "
            "If omitted, the first row is used only when the prediction CSV contains exactly one row."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output FASTA path. If omitted, a timestamped FASTA file is written to the current directory.",
    )
    parser.add_argument(
        "--max-skip-bp",
        type=int,
        default=DEFAULT_MAX_SKIP_BP,
        help="Maximum number of skipped backbone bases allowed for one branch substitution.",
    )
    parser.add_argument(
        "--max-alt-steps",
        type=int,
        default=DEFAULT_MAX_ALT_STEPS,
        help="Maximum number of graph steps followed along one alternative path.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=DEFAULT_FASTA_LINE_WIDTH,
        help="FASTA line width.",
    )
    parser.add_argument(
        "--chrom-name-style",
        choices=["as-is", "chm13"],
        default="as-is",
        help=(
            "Chromosome names used in the output FASTA. Use 'as-is' to keep GFA "
            "SN:Z names, or 'chm13' to convert known T2T-CHM13 RefSeq accessions "
            "such as NC_060925.1 to chr1."
        ),
    )
    parser.add_argument(
        "--chrom-map",
        default=None,
        help=(
            "Optional CSV/TSV file that maps GFA chromosome names to FASTA header names. "
            "Use columns 'source' and 'target', or use the first two columns. This "
            "mapping is applied after --chrom-name-style."
        ),
    )
    parser.add_argument(
        "--allow-unmapped-chroms",
        action="store_true",
        help=(
            "Keep chromosome names unchanged if they are not present in the selected "
            "chromosome-name mapping. By default, unmapped names raise an error when "
            "a mapping is requested."
        ),
    )
    return parser


def main() -> None:
    """Command-line entry point."""
    args = build_arg_parser().parse_args()

    reconstruct_anchored_genome(
        gfa_path=args.gfa,
        pred_csv_path=args.predictions,
        target_sample=args.sample,
        output_fasta=args.output,
        max_skip_bp=args.max_skip_bp,
        max_alt_steps=args.max_alt_steps,
        line_width=args.line_width,
        chrom_name_style=args.chrom_name_style,
        chrom_map_path=args.chrom_map,
        allow_unmapped_chroms=args.allow_unmapped_chroms,
    )


if __name__ == "__main__":
    main()
