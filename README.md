# PGpath

PGpath is a pangenome-based framework for recommending population-adapted linear reference genomes. Given paired-end sequencing reads from a target population, PGpath extracts population-level k-mer frequency features, predicts branch-node selections in a pangenome graph, and reconstructs a population-adapted linear reference genome in FASTA format.

PGpath is designed to translate pangenome diversity into a practical linear reference that can be directly used in standard read mapping and variant detection workflows.

---

## Overview

PGpath contains two usage modes:

1. **Direct use with a pretrained model**  
   Use `pgpath.py` to generate a PGpath-derived reference genome from paired-end FASTQ files.

2. **Retraining PGpath**  
   Use `train_pangenome3_refactored_v4.py` to train a new topology-aware branch-node prediction model, then export the selected k-mers and scaler statistics for inference.

---

## Repository structure

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


Main scripts
| Script                                 | Description                                                                                              |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `pgpath.py`                            | One-command PGpath pipeline for direct use.                                                              |
| `build_population_kmer_features.py`    | Counts selected k-mers from paired-end FASTQ files and builds a population-level k-mer frequency vector. |
| `inference_refactored_v2.py`           | Runs branch-node prediction using a trained PGpath model.                                                |
| `reconstruct_genome2_refactored_v3.py` | Reconstructs a PGpath-derived linear reference FASTA from predicted branch-node labels and a GFA graph.  |
| `train_pangenome3_refactored_v4.py`    | Trains the topology-aware multi-task branch-node prediction model.                                       |
| `extract_training_kmers_and_scaler.py` | Exports selected k-mers and StandardScaler statistics from the training feature matrix.                  |


Requirements
Python packages

PGpath requires Python 3.9 or later. The main Python dependencies are:
```bash
pip install numpy pandas scipy scikit-learn torch
```


Jellyfish

PGpath uses Jellyfish to count k-mers from FASTQ files.

Please install Jellyfish before running PGpath:


```bash
conda install -c bioconda jellyfish
```

or

```bash
sudo apt-get install jellyfish
```
Check installation:
```bash
jellyfish --version
```

Direct use

For most users, the recommended entry point is:


```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta
```

Required inputs

Only two arguments are required for direct use:

| Argument            | Description                                                            |
| ------------------- | ---------------------------------------------------------------------- |
| `-i`, `--input-dir` | Folder containing paired-end FASTQ files from the target population.   |
| `-o`, `--output`    | Output FASTA file for the PGpath-derived population-adapted reference. |

Default resource files

By default, pgpath.py expects the following resource files to be placed in the same directory as pgpath.py:

```bash
pgpath_selected_kmers.txt
pgpath_scaler_stats.csv
trained_model.pth
labels.csv
graph_20260520.gfa
```

These files can also be manually specified if needed:

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

FASTQ naming convention

PGpath automatically detects paired-end FASTQ files with names such as:

```bash
sample1_R1.fastq.gz
sample1_R2.fastq.gz
sample2_R1.fastq.gz
sample2_R2.fastq.gz
```

It also supports lane-split naming patterns such as:


```bash
sample1_L001_R1_001.fastq.gz
sample1_L001_R2_001.fastq.gz
sample1_L002_R1_001.fastq.gz
sample1_L002_R2_001.fastq.gz
```
If FASTQ files are stored in nested subdirectories, add:
```bash
--recursive
```
Example:
```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  --recursive
```

What pgpath.py does

The integrated pipeline performs three steps:

Step 1: Build population-level k-mer features

PGpath counts the selected k-mers from each paired-end sample using Jellyfish. For each sample, selected k-mer counts are normalized within that sample. The normalized sample-level vectors are then averaged to obtain one population-level k-mer frequency vector.

The intermediate output has the following format:
```bash
        kmer_1      kmer_2      kmer_3      ...
new_population  0.00057    0.00054    0.00086    ...
```

Step 2: Predict branch-node labels

The population-level k-mer feature vector is standardized using the training scaler statistics and passed into the trained PGpath model. The model predicts one branch-node label for each branch in the pangenome graph.

Step 3: Reconstruct the PGpath-derived reference

The predicted branch-node labels are used to reconstruct a population-adapted linear reference genome from the pangenome graph. The output is a FASTA file.

Useful options for direct use
Number of Jellyfish threads

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  --threads 32
```
Jellyfish hash size

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  --hash-size 4G
```

Use GPU for inference

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  --device cuda
```

Keep intermediate files

By default, intermediate files are removed after the final FASTA is generated. To keep them:

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  --keep-intermediate
```

This keeps the intermediate population feature CSV and predicted branch-node CSV.

Output chromosome names

If the GFA backbone is based on T2T-CHM13 RefSeq chromosome accessions, PGpath can convert names such as NC_060925.1 to chr1 in the output FASTA:
```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  --chrom-name-style chm13
```
This is the default behavior in pgpath.py.

To keep the original chromosome names from the GFA file:

```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta \
  --chrom-name-style as-is
```


Retraining PGpath

If you want to train a new PGpath model, start from train_pangenome3_refactored_v4.py.

Step 1: Train the topology-aware branch-node prediction model
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

| Argument            | Description                                                            |
| ------------------- | ---------------------------------------------------------------------- |
| `-i`, `--features`  | Training k-mer feature matrix.                                         |
| `-l`, `--labels`    | Branch-node label matrix.                                              |
| `-r`, `--relations` | Branch-node topology relation file with `source` and `target` columns. |
| `-o`, `--output`    | Output model file.                                                     |

The model is saved as a PyTorch .pth file.

Training log

During training, the script prints:

```bash
Epoch [1/200] | CE Loss: ... | Graph Loss: ... | Lambda*Graph: ... | Total Loss: ...
```
where:
| Term           | Meaning                                    |
| -------------- | ------------------------------------------ |
| `CE Loss`      | Branch-node prediction cross-entropy loss. |
| `Graph Loss`   | Topology-aware graph smoothness loss.      |
| `Lambda*Graph` | Weighted topology-aware loss.              |
| `Total Loss`   | Final training loss.                       |
Step 2: Export selected k-mers and scaler statistics

After training, export the selected k-mers and StandardScaler statistics from the training feature matrix:
```bash
python extract_training_kmers_and_scaler.py \
  -i features_rigorous_filtered_2005.csv \
  -k pgpath_selected_kmers.txt \
  -s pgpath_scaler_stats.csv
```

This generates:
```bash
pgpath_selected_kmers.txt
pgpath_scaler_stats.csv
```
These two files are required for inference when using the integrated pgpath.py pipeline.

Step 3: Build population-level k-mer features from new FASTQ files

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
```bash
new_population.csv
```
This file contains one row representing the averaged population-level k-mer frequency vector.

Step 4: Run branch-node inference
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
```bash
predicted_pangenome_paths_new_population.csv
```
This file contains predicted branch-node labels and is used by the reconstruction step.

Step 5: Reconstruct the PGpath-derived FASTA reference
```bash
python reconstruct_genome2_refactored_v3.py \
  -g graph_20260520.gfa \
  -p predicted_pangenome_paths_new_population.csv \
  -o PGpath_reference_new_population.fasta \
  --chrom-name-style chm13
```
Output:
```bash
PGpath_reference_new_population.fasta
```

Expected input file formats
Training feature matrix

The training feature matrix should be a CSV file with samples as rows and selected k-mers as columns:

```bash
sample_id,kmer_1,kmer_2,kmer_3,...
sample_1,0.00057,0.00054,0.00086,...
sample_2,0.00061,0.00049,0.00079,...
```
Label file

The label file should be a CSV file with samples as rows and branches as columns:
```bash
sample_id,branch_1,branch_2,branch_3,...
sample_1,s123,s456,s789,...
sample_2,s123,s457,s790,...
```
Relation file

The topology relation file should contain two columns:

```bash
source,target
s123,s456
s456,s789
...
```

New population feature file

The new population feature file used for inference should contain one or more population rows and the selected k-mers as columns:
```bash
sample_id,kmer_1,kmer_2,kmer_3,...
new_population,0.00057,0.00054,0.00086,...
```
Prediction file

The prediction file generated by inference contains predicted branch-node labels:
```bash
Sample_ID,branch_1,branch_2,branch_3,...
new_population,s123,s456,s789,...
```
Recommended direct-use workflow

For users who only want to generate a PGpath-derived reference from FASTQ files:
```bash
python pgpath.py \
  -i /path/to/fastq_folder \
  -o PGpath_reference_new_population.fasta
```

This command performs k-mer counting, population-level feature construction, branch-node inference, and FASTA reconstruction automatically.


Notes
1.pgpath.py assumes the pretrained model and resource files are available in the same directory as the script unless paths are explicitly provided.
2.Large files such as graph_20260520.gfa and trained_model.pth may be better distributed through Git LFS, Zenodo, Figshare, or GitHub Releases instead of being stored directly in the repository.
3.The output FASTA can be used as a linear reference genome in standard downstream workflows such as read mapping and variant detection.
4.If CUDA is available, inference can run on GPU by setting --device cuda or leaving --device auto.
6.Jellyfish must be installed and available in the system path unless a custom executable is provided with --jellyfish.

Citation
If you use PGpath in your work, please cite the corresponding PGpath manuscript.


