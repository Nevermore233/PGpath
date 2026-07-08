# PGpath

**PGpath** is a pangenome-based framework for recommending population-adapted linear reference genomes. Given paired-end sequencing reads from a target population, PGpath extracts population-level k-mer frequency features, predicts branch-node selections in a pangenome graph, and reconstructs a population-adapted linear reference genome in FASTA format.

> PGpath translates pangenome diversity into a practical linear reference that can be used directly in standard read mapping and variant detection workflows.

---

## Overview

PGpath supports two main usage modes:

1. **Direct use with a pretrained model**

   Use `pgpath.py` to generate a PGpath-derived population-adapted reference genome from paired-end FASTQ files.

2. **Retraining PGpath**

   Use `train_pangenome3_refactored_v4.py` to train a new topology-aware branch-node prediction model, then export selected k-mers and scaler statistics for inference.

---

## Repository Structure

```text
PGpath/
├── pgpath.py
├── build_population_kmer_features.py
├── inference_refactored_v2.py
├── reconstruct_genome2_refactored_v3.py
├── train_pangenome3_refactored_v4.py
├── extract_training_kmers_and_scaler.py
├── pgpath_selected_kmers.txt
├── pgpath_scaler_stats.csv
├── trained_model.pth
├── labels.csv
└── graph_20260520.gfa
```

## Main Scripts

| Script | Description |
| --- | --- |
| `pgpath.py` | One-command PGpath pipeline for direct use. |
| `build_population_kmer_features.py` | Counts selected k-mers from paired-end FASTQ files and builds a population-level k-mer frequency vector. |
| `inference_refactored_v2.py` | Runs branch-node prediction using a trained PGpath model. |
| `reconstruct_genome2_refactored_v3.py` | Reconstructs a PGpath-derived linear reference FASTA from predicted branch-node labels and a GFA graph. |
| `train_pangenome3_refactored_v4.py` | Trains the topology-aware multi-task branch-node prediction model. |
| `extract_training_kmers_and_scaler.py` | Exports selected k-mers and StandardScaler statistics from the training feature matrix. |

---

## Requirements

### Python Packages

PGpath requires **Python 3.9 or later**.

Install the main Python dependencies:

```bash
pip install numpy pandas scipy scikit-learn torch
```

### Jellyfish

PGpath uses **Jellyfish** to count k-mers from FASTQ files.

Install Jellyfish before running PGpath:

```bash
conda install -c bioconda jellyfish
```

Alternatively, on Debian/Ubuntu systems:

```bash
sudo apt-get install jellyfish
```

Check the installation:

```bash
jellyfish --version
```

---

## Quick Start: Direct Use

For most users, the recommended entry point is:

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta
```

This command automatically performs:

1. k-mer counting
2. population-level feature construction
3. branch-node inference
4. FASTA reconstruction

### Required Inputs

Only two arguments are required for direct use:

| Argument | Description |
| --- | --- |
| `-i`, `--input-dir` | Folder containing paired-end FASTQ files from the target population. |
| `-o`, `--output` | Output FASTA file for the PGpath-derived population-adapted reference. |

### Default Resource Files

By default, `pgpath.py` expects the following resource files to be placed in the same directory as `pgpath.py`:

```text
pgpath_selected_kmers.txt
pgpath_scaler_stats.csv
trained_model.pth
labels.csv
graph_20260520.gfa
```

These files can also be specified manually:

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  -k pgpath_selected_kmers.txt \
  --scaler-stats pgpath_scaler_stats.csv \
  -m trained_model.pth \
  -l labels.csv \
  -g graph_20260520.gfa
```

---

## FASTQ Naming Convention

PGpath automatically detects paired-end FASTQ files with names such as:

```text
sample1_R1.fastq.gz
sample1_R2.fastq.gz
sample2_R1.fastq.gz
sample2_R2.fastq.gz
```

PGpath also supports lane-split naming patterns such as:

```text
sample1_L001_R1_001.fastq.gz
sample1_L001_R2_001.fastq.gz
sample1_L002_R1_001.fastq.gz
sample1_L002_R2_001.fastq.gz
```

If FASTQ files are stored in nested subdirectories, add `--recursive`:

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  --recursive
```

---

## What `pgpath.py` Does

The integrated PGpath pipeline performs three major steps.

### Step 1: Build Population-Level k-mer Features

PGpath counts selected k-mers from each paired-end sample using Jellyfish. For each sample, selected k-mer counts are normalized within that sample. The normalized sample-level vectors are then averaged to obtain one population-level k-mer frequency vector.

The intermediate output has the following format:

```text
                kmer_1      kmer_2      kmer_3      ...
new_population  0.00057     0.00054     0.00086     ...
```

### Step 2: Predict Branch-Node Labels

The population-level k-mer feature vector is standardized using the training scaler statistics and passed into the trained PGpath model.

The model predicts one branch-node label for each branch in the pangenome graph.

### Step 3: Reconstruct the PGpath-Derived Reference

The predicted branch-node labels are used to reconstruct a population-adapted linear reference genome from the pangenome graph.

The final output is a FASTA file.

---

## Useful Options for Direct Use

### Set the Number of Jellyfish Threads

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  --threads 32
```

### Set the Jellyfish Hash Size

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  --hash-size 4G
```

### Use GPU for Inference

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  --device cuda
```

### Keep Intermediate Files

By default, intermediate files are removed after the final FASTA is generated.

To keep them:

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  --keep-intermediate
```

This keeps the intermediate population feature CSV and predicted branch-node CSV.

### Output Chromosome Names

If the GFA backbone is based on T2T-CHM13 RefSeq chromosome accessions, PGpath can convert names such as `NC_060925.1` to `chr1` in the output FASTA:

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  --chrom-name-style chm13
```

This is the default behavior in `pgpath.py`.

To keep the original chromosome names from the GFA file:

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  --chrom-name-style as-is
```

---

## Retraining PGpath

If you want to train a new PGpath model, start from `train_pangenome3_refactored_v4.py`.

### Step 1: Train the Topology-Aware Branch-Node Prediction Model

```bash
python train_pangenome3_refactored_v4.py \
  -i features_rigorous_filtered_2005.csv \
  -l labels.csv \
  -r label_relations.csv \
  -o trained_model.pth \
  --epochs 200 \
  --batch-size 64 \
  --lambda-graph 1e-4 \
  --device auto
```

Required inputs:

| Argument | Description |
| --- | --- |
| `-i`, `--features` | Training k-mer feature matrix. |
| `-l`, `--labels` | Branch-node label matrix. |
| `-r`, `--relations` | Branch-node topology relation file with `source` and `target` columns. |
| `-o`, `--output` | Output model file. |

The trained model is saved as a PyTorch `.pth` file.

### Training Log

During training, the script prints:

```text
Epoch [1/200] | CE Loss: ... | Graph Loss: ... | Lambda*Graph: ... | Total Loss: ...
```

| Term | Meaning |
| --- | --- |
| `CE Loss` | Branch-node prediction cross-entropy loss. |
| `Graph Loss` | Topology-aware graph smoothness loss. |
| `Lambda*Graph` | Weighted topology-aware loss. |
| `Total Loss` | Final training loss. |

### Step 2: Export Selected k-mers and Scaler Statistics

After training, export the selected k-mers and StandardScaler statistics from the training feature matrix:

```bash
python extract_training_kmers_and_scaler.py \
  -i features_rigorous_filtered_2005.csv \
  -k pgpath_selected_kmers.txt \
  -s pgpath_scaler_stats.csv
```

This generates:

```text
pgpath_selected_kmers.txt
pgpath_scaler_stats.csv
```

These two files are required for inference when using the integrated `pgpath.py` pipeline.

### Step 3: Build Population-Level k-mer Features from New FASTQ Files

This step can be run separately if you want to inspect the population-level feature vector before inference:

```bash
python build_population_kmer_features.py \
  -i /path/to/fastq_folder \
  -k pgpath_selected_kmers.txt \
  -o new_population.csv \
  -n new_population \
  --threads 16 \
  --hash-size 1G
```

Output:

```text
new_population.csv
```

This file contains one row representing the averaged population-level k-mer frequency vector.

### Step 4: Run Branch-Node Inference

```bash
python inference_refactored_v2.py \
  -m trained_model.pth \
  -i new_population.csv \
  -t pgpath_selected_kmers.txt \
  --scaler-stats pgpath_scaler_stats.csv \
  -l labels.csv \
  -o predicted_pangenome_paths_new_population.csv
```

Output:

```text
predicted_pangenome_paths_new_population.csv
```

This file contains predicted branch-node labels and is used by the reconstruction step.

### Step 5: Reconstruct the PGpath-Derived FASTA Reference

```bash
python reconstruct_genome2_refactored_v3.py \
  -g graph_20260520.gfa \
  -p predicted_pangenome_paths_new_population.csv \
  -o PGpath_reference_new_population.fasta \
  --chrom-name-style chm13
```

Output:

```text
PGpath_reference_new_population.fasta
```

---

## Expected Input File Formats

### Training Feature Matrix

The training feature matrix should be a CSV file with samples as rows and selected k-mers as columns:

```csv
sample_id,kmer_1,kmer_2,kmer_3,...
sample_1,0.00057,0.00054,0.00086,...
sample_2,0.00061,0.00049,0.00079,...
```

### Label File

The label file should be a CSV file with samples as rows and branches as columns:

```csv
sample_id,branch_1,branch_2,branch_3,...
sample_1,s123,s456,s789,...
sample_2,s123,s457,s790,...
```

### Relation File

The topology relation file should contain two columns:

```csv
source,target
s123,s456
s456,s789
...
```

### New Population Feature File

The new population feature file used for inference should contain one or more population rows and the selected k-mers as columns:

```csv
sample_id,kmer_1,kmer_2,kmer_3,...
new_population,0.00057,0.00054,0.00086,...
```

### Prediction File

The prediction file generated by inference contains predicted branch-node labels:

```csv
Sample_ID,branch_1,branch_2,branch_3,...
new_population,s123,s456,s789,...
```

---

## Recommended Direct-Use Workflow

For users who only want to generate a PGpath-derived reference from FASTQ files, run:

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta
```

This single command performs k-mer counting, population-level feature construction, branch-node inference, and FASTA reconstruction automatically.

---

## Notes

1. `pgpath.py` assumes the pretrained model and resource files are available in the same directory as the script unless paths are explicitly provided.
2. Large files such as `graph_20260520.gfa` and `trained_model.pth` may be better distributed through Git LFS, Zenodo, Figshare, or GitHub Releases instead of being stored directly in the repository.
3. The output FASTA can be used as a linear reference genome in standard downstream workflows such as read mapping and variant detection.
4. If CUDA is available, inference can run on GPU by setting `--device cuda` or leaving `--device auto`.
5. Jellyfish must be installed and available in the system path unless a custom executable is provided with `--jellyfish`.

---

## Citation

If you use PGpath in your work, please cite the corresponding PGpath manuscript.
