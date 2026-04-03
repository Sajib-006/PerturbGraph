# PerturbGraph

**PerturbGraph: Graph-based prediction of unseen gene perturbation responses from Perturb-seq data**

This repository provides a unified framework for predicting transcriptional responses to **unseen gene perturbations** using graph-based representations, biological priors, and machine learning models.

It supports classical regressors, deep learning models, and graph neural networks, along with biological feature enrichment (GO, pathways, STRING).

---

## 🔥 Key Features

- Predict perturbation effects for **unseen genes**
- Works with **single-cell Perturb-seq (.h5ad)** data
- Builds **pseudo-bulk delta expression**
- Supports multiple graph types:
  - STRING PPI
  - Co-expression
  - TF regulatory networks
- Optional biological features:
  - GO embeddings
  - Pathway embeddings
- Multiple model families:
  - Linear: Ridge, Lasso, ElasticNet
  - ML: Random Forest, KNN, XGBoost, LightGBM
  - Deep: MLP, Autoencoder
  - Graph: GCN, GraphSAGE, GAT
- Reproducible benchmark pipelines (ECCB experiments)

---

## 📦 Installation

### Option 1: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate perturbgraph
````

### Option 2: Pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 📁 Required Data

You need:

* Perturb-seq dataset (`.h5ad`)
* STRING network:

  * `protein.links`
  * `protein.info`
* (Optional) GO annotations (`.tsv`)
* (Optional) pathway file (`.gmt`)

---

## ⚡ Quick Start

Minimal run:

```bash
python stable_shift_bench.py \
  --h5ad data/K562.h5ad \
  --string_links data/9606.protein.links.txt \
  --string_info data/9606.protein.info.txt \
  --outdir results/test_run \
  --model_type ridge \
  --graph_type STRING_only \
  --use_bio_features \
  --seed 42
```

---

## 🚀 Full Example (with GO features)

```bash
python stable_shift_bench.py \
  --h5ad data/K562.h5ad \
  --string_links data/9606.protein.links.txt \
  --string_info data/9606.protein.info.txt \
  --go_file data/go_annotations.tsv \
  --outdir results/k562_gcn_go \
  --model_type gcn \
  --graph_type STRING_only \
  --use_bio_features \
  --use_go_features \
  --go_gene_col gene \
  --go_term_col go_id \
  --go_svd_dim 64 \
  --node2vec_dim 256 \
  --epochs 300 \
  --seed 42
```

---

## 🧪 Reproducing Paper Experiments

### K562 benchmark

```bash
bash run_benchmark_k562.sh
```

### Norman benchmark

```bash
bash run_benchmark_norman.sh
```

---

## 🧬 Preparing Norman Dataset

```bash
python prepare_norman_for_benchmark.py \
  --input data/norman_raw.h5ad \
  --output data/norman_processed.h5ad
```

This:

* Converts perturbation labels
* Removes combinatorial perturbations
* Keeps single-gene perturbations only

---

## 📊 Qualitative Analysis

```bash
python qualitative_analysis.py \
  --h5ad data/K562.h5ad \
  --string_links data/9606.protein.links.txt \
  --string_info data/9606.protein.info.txt \
  --go_file data/go_annotations.tsv \
  --outdir results/qualitative \
  --seed 42
```

Outputs:

* gene-level metrics
* pathway analysis
* top-k overlap metrics
* plots

---

## ⚙️ Main Arguments

### Core inputs

```
--h5ad
--string_links
--string_info
--outdir
```

### Model selection

```
--model_type
```

Options:

```
mean, knn, ridge, lasso, elasticnet, pls,
rf, xgb, lgbm,
mlp, ae, dae,
gcn, sage, gat
```

---

### Graph options

```
--graph_type
```

Options:

```
none
STRING_only
COEXPR_only
STRING_plus_COEXPR
TF_only
STRING_plus_TF
```

---

### Feature options

```
--use_bio_features
--use_go_features
--go_file
--use_pathway_features
--pathway_file
```

---

### Training

```
--epochs
--lr
--hidden_dim
--dropout
--weight_decay
--patience
--seed
```

---

### Node2Vec

```
--node2vec_dim
--n2v_walk_length
--n2v_num_walks
--n2v_window
--n2v_p
--n2v_q
```

---

## 📤 Outputs

Each run produces:

* `results.json` → metrics + config
* `summary.csv` → aggregated results
* feature files:

  * GO embeddings
  * pathway embeddings
* Node2Vec embeddings (cached)
* logs (if using scripts)

---

## 📦 Requirements

Main dependencies:

```
numpy
pandas
scipy
scikit-learn
scanpy
matplotlib
networkx
torch
torch-geometric
node2vec
xgboost
lightgbm
```

---

## 🧠 Notes

* `stable_shift_bench.py` → main script
* `stable_shift_bench_extended.py` → for running all baselines called in run_all_benchmark_replogle.sh
* shell scripts reproduce ECCB experiments
* notebooks are optional for visualization

---

## 📜 Citation

If you use this work, please cite:

```
Dip, Sajib Acharjee, and Liqing Zhang. "Predicting Unseen Gene Perturbation Response Using Graph Neural Networks with Biological Priors." bioRxiv (2026): 2026-03.
```

---

## 🧹 .gitignore

```gitignore
__pycache__/
*.pyc
.ipynb_checkpoints/
data/
results/
logs/
*.npy
*.npz
*.h5ad
```

