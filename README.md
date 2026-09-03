# PGpath

PGpath builds a population-level k-mer profile from paired-end FASTQ files,
predicts pangenome branch-node selections, and reconstructs a population-adapted
linear FASTA reference. The primary command is `pgpath.py`.

This package implements the manuscript's 1,967 selected 31-mer feature
space and backbone-guided reconstruction. It does not include ablation-study or
supplementary-experiment workflows.

## Contents

| File | Purpose |
| --- | --- |
| `pgpath.py` | One-command FASTQ-to-FASTA workflow. |
| `pgpath_kmer_profile.py` | Build the one-row population k-mer feature CSV. |
| `pgpath_infer.py` | Standardize features and predict branch-node labels. |
| `pgpath_reconstruct.py` | Build a linear FASTA from predictions and a GFA graph. |
| `pgpath_train.py` | Train the topology-regularized branch-node model. |
| `pgpath_prepare_features.py` | Export a selected k-mer list and scaler statistics from a training-only feature matrix. |
| `pgpath_selected_kmers.txt` | Selected 1,967 31-mers in model input order. |
| `features_rigorous_filtered_1967.csv` | The 1,967-feature matrix supplied with this package. |
| `pgpath_scaler_stats.csv` | Mean and scale values for the same 1,967 features, in identical order. |
| `label_relations.csv` | Branch-node source-consistency relations for model training. |

## Requirements

Use Python 3.9 or later. Install the Python dependencies:

```bash
pip install numpy pandas scipy scikit-learn torch
```

Install [Jellyfish](https://github.com/gmarcais/Jellyfish) separately and make
the executable available on `PATH`, or provide it through `--jellyfish`.

## Required External Resources

The three large inference resources are not bundled in this directory. Download
them from the project release location and place them beside `pgpath.py`, or pass
their paths explicitly:

| Resource | Expected file name | Purpose |
| --- | --- | --- |
| Trained model | `trained_model.pth` | Weights compatible with the 1,967-feature space. |
| Branch labels | `labels.csv` | Candidate-node class mapping for each branch. |
| Pangenome graph | `pangenome_graph_default.gfa` | GFA graph with T2T-CHM13 primary backbone tags. |

The selected k-mer list and scaler statistics are a paired resource. Their
`kmer` values and order must be identical. The bundled `pgpath_scaler_stats.csv`
has been restricted to the 1,967 columns in
`features_rigorous_filtered_1967.csv` and ordered to match
`pgpath_selected_kmers.txt`. The retained mean and scale values are unchanged
from the supplied scaler resource. PGpath checks the pairing before prediction
and stops rather than applying an incorrect normalization.

## Direct Use

With compatible resources in the same directory, run:

```bash
python pgpath.py \
  --input-dir /path/to/fastq_folder \
  --output PGpath_reference.fasta
```

The equivalent explicit-resource command is:

```bash
python pgpath.py \
  --input-dir /path/to/fastq_folder \
  --output PGpath_reference.fasta \
  --kmers pgpath_selected_kmers.txt \
  --scaler-stats pgpath_scaler_stats.csv \
  --model trained_model.pth \
  --labels labels.csv \
  --gfa pangenome_graph_default.gfa \
  --population-name new_population \
  --threads 16 \
  --hash-size 1G \
  --device auto \
  --chrom-name-style chm13
```

`pgpath.py` writes a population feature CSV and prediction CSV to a timestamped
working directory, then removes them after successful or failed execution unless
`--keep-intermediate` is provided. Use `--work-dir`,
`--population-feature-output`, or `--prediction-output` to control their paths.
The final output remains the FASTA passed to `--output`.

### FASTQ naming

Supported suffixes are `.fastq`, `.fq`, `.fastq.gz`, and `.fq.gz`. Pair names
must clearly contain mate identifiers separated from the sample name by `_`,
`-`, or `.`. Examples:

```text
sample_R1.fastq.gz      sample_R2.fastq.gz
sample_1.fq.gz          sample_2.fq.gz
sample_L001_R1_001.fq   sample_L001_R2_001.fq
```

Lane-split files for the same sample are combined. Add `--recursive` when FASTQ
files are stored in subdirectories. PGpath retains the input feature and FASTA
formats; sample-level vectors are normalized by their selected-k-mer totals and
then averaged into one population vector.

## Independent Commands

Build only the feature vector:

```bash
python pgpath_kmer_profile.py \
  --input-dir /path/to/fastq_folder \
  --kmers pgpath_selected_kmers.txt \
  --output population.features.csv \
  --population-name new_population
```

Run inference from a prepared feature CSV:

```bash
python pgpath_infer.py \
  --model trained_model.pth \
  --input population.features.csv \
  --training-resource pgpath_selected_kmers.txt \
  --scaler-stats pgpath_scaler_stats.csv \
  --labels labels.csv \
  --output population.predicted_paths.csv \
  --confidence-output population.confidence.csv
```

The prediction CSV has population names in the first column and branch names in
the remaining columns. The optional confidence CSV has the same shape. The
inference script rejects duplicate sample IDs, duplicate feature names,
non-numeric values, incompatible resource orders, and model-shape mismatches.
Classes that only exist as model-output padding for a smaller branch are masked
before decoding.

Reconstruct from predicted branch nodes:

```bash
python pgpath_reconstruct.py \
  --gfa pangenome_graph_default.gfa \
  --predictions population.predicted_paths.csv \
  --sample new_population \
  --output PGpath_reference.fasta \
  --max-skip-bp 500000 \
  --max-alt-steps 1000 \
  --chrom-name-style chm13
```

During reconstruction, PGpath iterates through the ordered primary backbone.
It accepts a predicted non-primary alternative only if a directed path from the
current backbone node reaches a downstream backbone node on the same chromosome.
Unselected connector nodes are allowed, and predicted successors are preferred
for nested alternatives. Cycles, ambiguous connector choices, missing segment
sequences, cross-chromosome or upstream re-entry, excessive skipped backbone,
and paths longer than the step limit fall back to the backbone. The 500,000 bp
and 1,000-step limits are graph-safety limits, not biological thresholds. FASTA
output is written atomically to avoid leaving a partial file after a failure.

## Training and Reproducibility

`pgpath_train.py` expects **only the already prepared training partition**. It
does not create simulated populations, perform feature selection, or split a
combined training/validation/test matrix. This is intentional: before population
construction, simulated sources must be partitioned into disjoint 60% training,
20% validation, and 20% test source pools. Construct populations independently
within each pool. Perform every data-dependent filter, selected-k-mer export, and
standardization fit with the training populations only. Use the validation set to
choose hyperparameters such as `--lambda-graph`, then train the final model using
the chosen configuration; reserve the held-out test pool for final evaluation.

The default training configuration matches the manuscript: 500 epochs,
batch size 64, dropout 0.3, Adam learning rate `1e-3`, weight decay `1e-5`,
ReduceLROnPlateau factor 0.5 with patience 10, seed 42, default PyTorch
initialization, and no early stopping.

```bash
python pgpath_train.py \
  --features training_features_1967.csv \
  --labels training_labels.csv \
  --relations label_relations.csv \
  --output trained_model.pth \
  --epochs 500 \
  --batch-size 64 \
  --lambda-graph 1e-4 \
  --seed 42 \
  --device auto
```

After finalizing the training-only feature matrix, export paired inference
resources:

```bash
python pgpath_prepare_features.py \
  --input training_features_1967.csv \
  --kmers-output pgpath_selected_kmers.txt \
  --scaler-output pgpath_scaler_stats.csv
```

Do not supply validation, test, or pooled data to `pgpath_prepare_features.py`:
it fits `StandardScaler` to every row supplied to it. This explicit separation
prevents information leakage into the inference normalization parameters.

## Notes

- The bundled selected list is 1,967 31-mers, the manuscript default.
- The topology relation file regularizes branch predictions during training. It
  is distinct from graph-connectivity validation during reconstruction.
- Runtime and memory depend on hardware, storage, Jellyfish configuration,
  graph size, and input depth. See the supplement for benchmark context.
