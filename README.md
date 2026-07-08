# PGpath: Pangenome Subpath-Based Recommendation of Population-Adapted Linear Reference Genomes

**PGpath** is a pangenome-based framework for generating population-adapted linear reference genomes. Given paired-end sequencing reads from a target population, PGpath extracts population-level k-mer frequency features, predicts branch-node selections in a pangenome graph, and reconstructs a linear reference genome from the selected pangenome subpath.

> The generated reference preserves population-relevant pangenome sequences while remaining compatible with standard linear-reference workflows, including read mapping and variant detection.

---

## 1. Overview

PGpath supports two main use cases:

- **Direct reference recommendation with a pretrained model**
  - Use `pgpath.py`.
  - Input: a folder containing paired-end FASTQ files.
  - Output: a PGpath-derived population-adapted FASTA reference.

- **Model retraining**
  - Start from `train_pangenome3_refactored_v4.py`.
  - Input: training k-mer features, branch-node labels, and branch-node topological relations.
  - Output: a trained PGpath model and inference resources.

---

## 2. Repository Structure

```text
PGpath/
|-- pgpath.py
|-- build_population_kmer_features.py
|-- inference_refactored_v2.py
|-- reconstruct_genome2_refactored_v3.py
|-- train_pangenome3_refactored_v4.py
|-- extract_training_kmers_and_scaler.py
|-- pgpath_selected_kmers.txt
|-- pgpath_scaler_stats.csv
|-- trained_model.pth
|-- labels.csv
`-- graph_20260520.gfa
```

### 2.1 Core Scripts

| Script | Function |
| --- | --- |
| `pgpath.py` | One-command PGpath pipeline for direct use. |
| `build_population_kmer_features.py` | Counts selected k-mers from paired-end FASTQ files and builds population-level k-mer frequency features. |
| `inference_refactored_v2.py` | Predicts branch-node labels using a trained PGpath model. |
| `reconstruct_genome2_refactored_v3.py` | Reconstructs a PGpath-derived linear FASTA reference from predicted branch-node labels and a GFA graph. |
| `train_pangenome3_refactored_v4.py` | Trains the topology-aware multi-task branch-node prediction model. |
| `extract_training_kmers_and_scaler.py` | Exports selected k-mers and StandardScaler statistics from the training feature matrix. |

### 2.2 Default Resource Files

For direct use, the following resource files should be placed in the same directory as `pgpath.py`:

| File | Description |
| --- | --- |
| `pgpath_selected_kmers.txt` | Selected k-mer list used by PGpath. |
| `pgpath_scaler_stats.csv` | StandardScaler statistics exported from the training feature matrix. |
| `trained_model.pth` | Trained PGpath model weights. |
| `labels.csv` | Branch-node label file used to rebuild label mappings during inference. |
| `graph_20260520.gfa` | Pangenome graph used for reference reconstruction. |

> Large files such as `trained_model.pth` and `graph_20260520.gfa` can be distributed through GitHub Releases, Git LFS, Zenodo, Figshare, or another external file-hosting service.

---

## 3. Installation

### 3.1 Python Dependencies

PGpath requires **Python 3.9 or later**.

Install the required Python packages:

```bash
pip install numpy pandas scipy scikit-learn torch
```

### 3.2 Jellyfish

PGpath uses **Jellyfish** for k-mer counting.

Install Jellyfish with Conda:

```bash
conda install -c bioconda jellyfish
```

Alternatively, install Jellyfish with `apt`:

```bash
sudo apt-get install jellyfish
```

Check whether Jellyfish is available:

```bash
jellyfish --version
```

---

## 4. Direct Use with a Pretrained Model

### 4.1 Minimal Command

For most users, only two arguments are required:

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta
```

### 4.2 Required Arguments

| Argument | Description |
| --- | --- |
| `-i`, `--input-dir` | Folder containing paired-end FASTQ files from the target population. |
| `-o`, `--output` | Output FASTA file for the PGpath-derived population-adapted reference. |

### 4.3 Default Parameters

The minimal command uses the following default resources and settings:

```text
-k pgpath_selected_kmers.txt
--scaler-stats pgpath_scaler_stats.csv
-m trained_model.pth
-l labels.csv
-g graph_20260520.gfa
-n new_population
--threads 16
--hash-size 1G
--device auto
--chrom-name-style chm13
```

### 4.4 Full Command with Explicit Resources

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  -k pgpath_selected_kmers.txt \
  --scaler-stats pgpath_scaler_stats.csv \
  -m trained_model.pth \
  -l labels.csv \
  -g graph_20260520.gfa \
  -n new_population \
  --threads 16 \
  --hash-size 1G \
  --device auto \
  --chrom-name-style chm13
```

---

## 5. FASTQ Input Requirements

### 5.1 Supported File Extensions

PGpath recognizes the following FASTQ file extensions:

```text
.fastq.gz
.fq.gz
.fastq
.fq
```

### 5.2 Supported Paired-End Naming Formats

PGpath identifies paired-end samples from file names. The mate identifier must be clearly separated from the sample name by `_`, `-`, or `.`.

Supported examples:

```text
sample1_R1.fastq.gz
sample1_R2.fastq.gz
sample1_1.fq.gz
sample1_2.fq.gz
sample1-L001-R1-001.fastq.gz
sample1-L001-R2-001.fastq.gz
sample1_L001_R1_001.fastq.gz
sample1_L001_R2_001.fastq.gz
sample1_L002_R1_001.fastq.gz
sample1_L002_R2_001.fastq.gz
sim_HG002_3x_sample1_1.fq
sim_HG002_3x_sample1_2.fq
```

The last example is parsed as:

```text
sample name: sim_HG002_3x_sample1
mate 1:      sim_HG002_3x_sample1_1.fq
mate 2:      sim_HG002_3x_sample1_2.fq
```

### 5.3 Unsupported Ambiguous Naming

The following naming pattern is not recommended:

```text
sim_HG002_3x_sample11.fq
sim_HG002_3x_sample12.fq
```

This pattern is ambiguous because `sample11` may represent either `sample1` mate 1 or sample number 11.

Rename such files before running PGpath:

```text
sim_HG002_3x_sample1_1.fq
sim_HG002_3x_sample1_2.fq
```

or:

```text
sim_HG002_3x_sample1_R1.fq
sim_HG002_3x_sample1_R2.fq
```

### 5.4 Multiple Samples in One Folder

A folder may contain multiple paired-end samples:

```text
sample1_R1.fastq.gz
sample1_R2.fastq.gz
sample2_R1.fastq.gz
sample2_R2.fastq.gz
sample3_R1.fastq.gz
sample3_R2.fastq.gz
```

PGpath counts selected k-mers for each sample separately, normalizes the selected k-mer counts within each sample, and averages all sample-level vectors to obtain one population-level k-mer frequency vector.

### 5.5 Lane-Split FASTQ Files

Lane-split files are grouped by sample name:

```text
sample1_L001_R1_001.fastq.gz
sample1_L001_R2_001.fastq.gz
sample1_L002_R1_001.fastq.gz
sample1_L002_R2_001.fastq.gz
```

All R1 and R2 files belonging to the same sample are jointly used for k-mer counting.

### 5.6 Recursive Search

By default, PGpath searches FASTQ files only in the provided folder. Use `--recursive` if FASTQ files are stored in nested subdirectories:

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  --recursive
```

---

## 6. Output Files

### 6.1 Final Output

The main output is a FASTA file:

```text
PGpath_reference_new_population.fasta
```

This FASTA file is the PGpath-derived population-adapted linear reference genome.

### 6.2 Intermediate Outputs

By default, intermediate files are removed after the final FASTA is generated. Use `--keep-intermediate` to retain them:

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  --keep-intermediate
```

Intermediate files include:

| File | Description |
| --- | --- |
| `new_population.features.csv` | Population-level normalized k-mer frequency vector. |
| `new_population.predicted_paths.csv` | Predicted branch-node labels for reference reconstruction. |

---

## 7. Direct-Use Options

### 7.1 Number of Threads for Jellyfish

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  --threads 32
```

### 7.2 Jellyfish Hash Size

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  --hash-size 4G
```

### 7.3 Device for Model Inference

Use automatic device selection:

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  --device auto
```

Force GPU inference:

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  --device cuda
```

Force CPU inference:

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  --device cpu
```

### 7.4 Chromosome Name Style

When the GFA backbone uses T2T-CHM13 RefSeq accessions, PGpath can convert chromosome names such as `NC_060925.1` to `chr1` in the output FASTA.

Default behavior:

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  --chrom-name-style chm13
```

Keep chromosome names exactly as stored in the GFA file:

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  --chrom-name-style as-is
```

Use a custom chromosome-name mapping file:

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  --chrom-map chrom_name_map.csv
```

The mapping file should contain `source` and `target` columns. If these column names are not available, PGpath uses the first two columns as source and target names:

```csv
source,target
NC_060925.1,chr1
NC_060926.1,chr2
NC_060927.1,chr3
```

---

## 8. Internal Workflow of `pgpath.py`

The one-command pipeline performs three steps.

### 8.1 Step 1: Population-Level k-mer Feature Construction

`build_population_kmer_features.py` counts selected k-mers from each paired-end sample using Jellyfish. For each sample, selected k-mer counts are normalized by the total count of selected k-mers in that sample. Normalized sample-level vectors are averaged across all samples.

The resulting feature file has one row:

```csv
sample_id,kmer_1,kmer_2,kmer_3,...
new_population,0.00057,0.00054,0.00086,...
```

### 8.2 Step 2: Branch-Node Inference

`inference_refactored_v2.py` standardizes the population-level k-mer feature vector using `pgpath_scaler_stats.csv` and predicts branch-node labels with the trained PGpath model.

The prediction file has the following format:

```csv
Sample_ID,branch_1,branch_2,branch_3,...
new_population,s123,s456,s789,...
```

### 8.3 Step 3: Reference Reconstruction

`reconstruct_genome2_refactored_v3.py` traverses the primary backbone nodes in the GFA graph and inserts predicted non-backbone branch paths only when they connect from the current backbone node and rejoin a downstream backbone node on the same chromosome.

The final output is a linear FASTA reference.

---

## 9. Model Retraining

Use this section only when training a new PGpath model.

### 9.1 Required Training Files

| File | Description |
| --- | --- |
| `features_rigorous_filtered_2005.csv` | Training k-mer feature matrix. |
| `labels.csv` | Branch-node label matrix. |
| `label_relations.csv` | Branch-node topological relation file. |

### 9.2 Train a New PGpath Model

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

### 9.3 Training Log

During training, the script reports:

```text
Epoch [1/200] | CE Loss: ... | Graph Loss: ... | Lambda*Graph: ... | Total Loss: ...
```

| Term | Description |
| --- | --- |
| `CE Loss` | Cross-entropy loss for branch-node prediction. |
| `Graph Loss` | Topology-aware graph regularization loss. |
| `Lambda*Graph` | Weighted topology-aware loss. |
| `Total Loss` | Sum of cross-entropy loss and weighted topology-aware loss. |

### 9.4 Export Selected k-mers and Scaler Statistics

After training, export the selected k-mer list and scaler statistics from the training feature matrix:

```bash
python extract_training_kmers_and_scaler.py \
  -i features_rigorous_filtered_2005.csv \
  -k pgpath_selected_kmers.txt \
  -s pgpath_scaler_stats.csv
```

Generated files:

```text
pgpath_selected_kmers.txt
pgpath_scaler_stats.csv
```

These files are required for downstream inference.

---

## 10. Running Individual Modules

Although `pgpath.py` is recommended for direct use, each module can also be executed separately.

### 10.1 Build Population-Level k-mer Features

```bash
python build_population_kmer_features.py \
  -i /path/to/fastq_folder \
  -k pgpath_selected_kmers.txt \
  -o new_population.csv \
  -n new_population \
  --threads 16 \
  --hash-size 1G
```

### 10.2 Run Branch-Node Inference

```bash
python inference_refactored_v2.py \
  -m trained_model.pth \
  -i new_population.csv \
  -t pgpath_selected_kmers.txt \
  --scaler-stats pgpath_scaler_stats.csv \
  -l labels.csv \
  -o predicted_pangenome_paths_new_population.csv
```

### 10.3 Reconstruct a PGpath-Derived FASTA Reference

```bash
python reconstruct_genome2_refactored_v3.py \
  -g graph_20260520.gfa \
  -p predicted_pangenome_paths_new_population.csv \
  -o PGpath_reference_new_population.fasta \
  --chrom-name-style chm13
```

---

## 11. Input File Formats

### 11.1 Training Feature Matrix

```csv
sample_id,kmer_1,kmer_2,kmer_3
sample_1,0.00057,0.00054,0.00086
sample_2,0.00061,0.00049,0.00079
```

Rows correspond to training samples or simulated populations. Columns correspond to selected k-mers.

### 11.2 Branch-Node Label Matrix

```csv
sample_id,branch_1,branch_2,branch_3
sample_1,s123,s456,s789
sample_2,s123,s457,s790
```

Rows should match the training feature matrix. Columns correspond to pangenome graph branches. Values correspond to selected node IDs.

### 11.3 Branch-Node Relation File

```csv
source,target
s123,s456
s456,s789
```

The relation file defines topological associations between branch nodes. The training script uses these relations to construct the topology-aware regularization term.

### 11.4 Selected k-mer File

```text
ACCACGCCTGGCTAATTTTTGTATTTTTAGT
CAACAGGTGCTGGAGAGGATGTGGAGAAATA
CACTATTCACAATAGCAAAGACTTGGAACCA
```

The selected k-mer file contains one selected k-mer per line.

### 11.5 Scaler Statistics File

```csv
kmer,mean,scale
ACCACGCCTGGCTAATTTTTGTATTTTTAGT,0.00043,0.00012
CAACAGGTGCTGGAGAGGATGTGGAGAAATA,0.00051,0.00015
```

The inference script uses these statistics to standardize new population k-mer features consistently with model training.

---

## 12. Troubleshooting

### 12.1 No Paired-End FASTQ Samples Were Found

Check whether file names contain explicit mate identifiers such as `_R1`, `_R2`, `_1`, or `_2`.

Recommended:

```text
sample1_R1.fastq.gz
sample1_R2.fastq.gz
```

Not recommended:

```text
sample11.fq
sample12.fq
```

### 12.2 Jellyfish Executable Was Not Found

Install Jellyfish or provide its path manually:

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  --jellyfish /path/to/jellyfish
```

### 12.3 CUDA Was Requested but Unavailable

Use automatic device selection or CPU mode:

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  --device auto
```

or:

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  --device cpu
```

### 12.4 Missing Default Resource Files

Make sure the following files are in the same directory as `pgpath.py`:

```text
pgpath_selected_kmers.txt
pgpath_scaler_stats.csv
trained_model.pth
labels.csv
graph_20260520.gfa
```

Alternatively, provide explicit paths with `-k`, `--scaler-stats`, `-m`, `-l`, and `-g`.

---

## 13. Citation

If you use PGpath in your work, please cite the corresponding PGpath manuscript.
