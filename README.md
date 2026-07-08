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

For **direct use**, the following resource files should be placed in the same directory as `pgpath.py`:

| File | Description |
| --- | --- |
| `pgpath_selected_kmers.txt` | Selected k-mer list used by PGpath. |
| `pgpath_scaler_stats.csv` | StandardScaler statistics exported from the training feature matrix. |
| `trained_model.pth` | Trained PGpath model weights. |
| `labels.csv` | Branch-node label file used to rebuild label mappings during inference. |
| `graph_20260520.gfa` | Pangenome graph used for reference reconstruction. |

> Large files such as `trained_model.pth` and `graph_20260520.gfa` can be distributed through GitHub Releases, Git LFS, Zenodo, Figshare, or another external file-hosting service.

---

## 3. Requirements

### 3.1 Python Dependencies

PGpath requires **Python 3.9 or later**. Install the required Python packages:

```bash
pip install numpy pandas scipy scikit-learn torch
```

### 3.2 Jellyfish

PGpath uses **Jellyfish** for k-mer counting.Install Jellyfish with Conda:

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

### 4.1 Required Arguments

For most users, only two arguments are required:

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta
```

| Argument | Description |
| --- | --- |
| `-i`, `--input-dir` | Folder containing paired-end FASTQ files from the target population. |
| `-o`, `--output` | Output FASTA file for the PGpath-derived population-adapted reference. |

Full command with explicit resources

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

| Argument | Requirement or default | Description |
|---|---|---|
| `-i`, `--input-dir` | Required | Directory containing paired-end FASTQ files from the target population. |
| `-o`, `--output` | Required | Output FASTA file for the PGpath-derived population-adapted linear reference genome. |
| `-k`, `--kmers` | `pgpath_selected_kmers.txt` | Text file containing selected PGpath k-mers, one k-mer per line. |
| `--scaler-stats` | `pgpath_scaler_stats.csv` | CSV file containing StandardScaler statistics exported from the training feature matrix. These statistics ensure that new population k-mer features are normalized consistently with model training. |
| `-m`, `--model` | `trained_model.pth` | Trained PGpath model weights used for branch-node prediction. |
| `-l`, `--labels` | `labels.csv` | Branch-node label file used to rebuild the mapping between model output classes and original pangenome graph node IDs. |
| `-g`, `--gfa` | `graph_20260520.gfa` | Input pangenome graph in GFA format. PGpath reconstructs the final linear reference from this graph and the predicted branch-node labels. |
| `-n`, `--population-name` | `new_population` | Population name used as the row identifier in intermediate feature and prediction files. |
| `--threads` | `16` | Number of threads used by Jellyfish for k-mer counting. |
| `--hash-size` | `1G` | Jellyfish hash size for k-mer counting. Increase this value for large sequencing datasets if needed. |
| `--device` | `auto` | Device used for neural network inference. Options are `auto`, `cpu`, and `cuda`. With `auto`, PGpath uses CUDA if available, otherwise CPU. |
| `--chrom-name-style` | `chm13` | Chromosome naming style for the output FASTA. Use `chm13` to convert known T2T-CHM13 RefSeq accessions such as such as `NC_060925.1` to `chr1`. Use `as-is` to keep chromosome names from the GFA file unchanged. |

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


### 4.2 Supported FASTQ File Extensions

A folder may contain multiple paired-end sample:
```text
sample1_R1.fastq.gz
sample1_R2.fastq.gz
sample2_R1.fastq.gz
sample2_R2.fastq.gz
sample3_R1.fastq.gz
sample3_R2.fastq.gz
```

PGpath recognizes the following FASTQ file extensions:

```text
.fastq.gz
.fq.gz
.fastq
.fq
```
PGpath identifies paired-end samples from file names. The mate identifier must be clearly separated from the sample name by `_`, `-`, or `.`.

Supported examples:

```text
sample1_R1.fastq.gz
sample1_R2.fastq.gz
sample1_1.fq.gz
sample1_2.fq.gz
sample1-L001**-**R1-001.fastq.gz
sample1-L001**-**R2-001.fastq.gz
sample1_L001**_**R1_001.fastq.gz
sample1_L001**_**R2_001.fastq.gz
sample1_L002**_**R1_001.fastq.gz
sample1_L002**_**R2_001.fastq.gz
```
By default, PGpath searches FASTQ files only in the provided folder. Use `--recursive` if FASTQ files are stored in nested subdirectories:

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  --recursive
```
---

### 4.2 Output Files

The main output is a FASTA file:

```text
PGpath_reference_new_population.fasta
```

This FASTA file is the PGpath-derived population-adapted linear reference genome. By default, intermediate files are removed after the final FASTA is generated. Use `--keep-intermediate` to retain them:

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


## 5. Internal Workflow of `pgpath.py`

The one-command pipeline performs three steps.

### 5.1 Step 1: Population-Level k-mer Feature Construction

`build_population_kmer_features.py` counts selected k-mers from each paired-end sample using Jellyfish. For each sample, selected k-mer counts are normalized by the total count of selected k-mers in that sample. Normalized sample-level vectors are averaged across all samples.

The resulting feature file has one row:

```csv
sample_id,kmer_1,kmer_2,kmer_3,...
new_population,0.00057,0.00054,0.00086,...
```

### 5.2 Step 2: Branch-Node Inference

`inference_refactored_v2.py` standardizes the population-level k-mer feature vector using `pgpath_scaler_stats.csv` and predicts branch-node labels with the trained PGpath model.

The prediction file has the following format:

```csv
Sample_ID,branch_1,branch_2,branch_3,...
new_population,s123,s456,s789,...
```

### 5.3 Step 3: Reference Reconstruction

`reconstruct_genome2_refactored_v3.py` traverses the primary backbone nodes in the GFA graph and inserts predicted non-backbone branch paths only when they connect from the current backbone node and rejoin a downstream backbone node on the same chromosome.

The final output is a linear FASTA reference.

---

## 6. Model Retraining

Use this section only when training a new PGpath model.

### 6.1 Required Training Files

| File | Description |
| --- | --- |
| `features_rigorous_filtered_2005.csv` | Training k-mer feature matrix. |
| `labels.csv` | Branch-node label matrix. |
| `label_relations.csv` | Branch-node topological relation file. |

### 6.2 Train a New PGpath Model

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

### 6.3 Training Log

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

### 6.4 Export Selected k-mers and Scaler Statistics

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

## 7. Running Individual Modules

Although `pgpath.py` is recommended for direct use, each module can also be executed separately.

### 7.1 Build Population-Level k-mer Features

```bash
python build_population_kmer_features.py \
  -i /path/to/fastq_folder \
  -k pgpath_selected_kmers.txt \
  -o new_population.csv \
  -n new_population \
  --threads 16 \
  --hash-size 1G
```

### 7.2 Run Branch-Node Inference

```bash
python inference_refactored_v2.py \
  -m trained_model.pth \
  -i new_population.csv \
  -t pgpath_selected_kmers.txt \
  --scaler-stats pgpath_scaler_stats.csv \
  -l labels.csv \
  -o predicted_pangenome_paths_new_population.csv
```

### 7.3 Reconstruct a PGpath-Derived FASTA Reference

```bash
python reconstruct_genome2_refactored_v3.py \
  -g graph_20260520.gfa \
  -p predicted_pangenome_paths_new_population.csv \
  -o PGpath_reference_new_population.fasta \
  --chrom-name-style chm13
```

---

## 8. Input File Formats

### 8.1 Training Feature Matrix

```csv
sample_id,kmer_1,kmer_2,kmer_3
sample_1,0.00057,0.00054,0.00086
sample_2,0.00061,0.00049,0.00079
```

Rows correspond to training samples or simulated populations. Columns correspond to selected k-mers.

### 8.2 Branch-Node Label Matrix

```csv
sample_id,branch_1,branch_2,branch_3
sample_1,s123,s456,s789
sample_2,s123,s457,s790
```

Rows should match the training feature matrix. Columns correspond to pangenome graph branches. Values correspond to selected node IDs.

### 8.3 Branch-Node Relation File

```csv
source,target
s123,s456
s456,s789
```

The relation file defines topological associations between branch nodes. The training script uses these relations to construct the topology-aware regularization term.

### 8.4 Selected k-mer File

```text
ACCACGCCTGGCTAATTTTTGTATTTTTAGT
CAACAGGTGCTGGAGAGGATGTGGAGAAATA
CACTATTCACAATAGCAAAGACTTGGAACCA
```

The selected k-mer file contains one selected k-mer per line.

### 8.5 Scaler Statistics File

```csv
kmer,mean,scale
ACCACGCCTGGCTAATTTTTGTATTTTTAGT,0.00043,0.00012
CAACAGGTGCTGGAGAGGATGTGGAGAAATA,0.00051,0.00015
```

The inference script uses these statistics to standardize new population k-mer features consistently with model training.

---

## 9. Troubleshooting

### 9.1 No Paired-End FASTQ Samples Were Found

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

### 9.2 Jellyfish Executable Was Not Found

Install Jellyfish or provide its path manually:

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  --jellyfish /path/to/jellyfish
```

### 9.3 CUDA Was Requested but Unavailable

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

### 9.4 Missing Default Resource Files

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

## 10. Citation

If you use PGpath in your work, please cite the corresponding PGpath manuscript.
