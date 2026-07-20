# SSE-DDI

**Selective Substructure Encoding with Bond-Centered Molecular Representations for Drug–Drug Interaction Prediction**

SSE-DDI is a molecular structure-driven framework for drug–drug interaction prediction. It represents each molecule with directed bond states, performs non-backtracking substructure encoding, and combines molecular representations with SMILES-derived structural similarity profiles.


## Overview

Given a drug pair and a candidate interaction type, SSE-DDI predicts whether the interaction is present.

The released pipeline includes:

- molecular graph construction from SMILES;
- directed bond-state and non-backtracking line-graph preprocessing;
- Morgan-fingerprint Tanimoto similarity profiles;
- transductive and inductive data splits;
- fixed 1:1 positive/negative sampling;
- training and evaluation over five random seeds;
- ACC, AUC, F1, precision, recall, and AP reporting;
- reproducibility checks and metadata export.

Supported datasets:

- **DrugBank**
- **TwoSIDES**

---

## Method highlights

### Bond-centered molecular representation

Each undirected chemical bond is expanded into two directed bond states:

```text
u -> v
v -> u
```

The directed line graph allows transitions of the form:

```text
(u, v) -> (v, w)
```

subject to:

```text
w != u
```

This prevents immediate backtracking and supports directional substructure propagation.

### Structural similarity profile

For each drug, the preprocessing pipeline computes a Morgan fingerprint and its Tanimoto similarity to all drugs in the benchmark dictionary.

The refined similarity graph applies:

1. Top-K candidate selection;
2. adaptive thresholding;
3. minimum-degree backfilling;
4. Mutual-KNN filtering;
5. maximum-degree capping;
6. a second backfilling step.

The adaptive threshold is:

```text
tau = max(tau_q, tau_mu)
```

where:

```text
tau_mu = mean + lambda * population_standard_deviation
```

### Transductive and inductive evaluation

The repository supports:

- **Transductive:** positive triplets are split into 60% train, 20% validation, and 20% test.
- **Inductive S1:** exactly one drug in the pair is unseen during training.
- **Inductive S2:** both drugs are unseen during training.

Unseen drugs remain available through their SMILES-derived molecular structures and similarity profiles.

---

## Repository structure

```text
SSE-DDI/
├── data_pre.py
├── dataset.py
├── model.py
├── train.py
├── metrics.py
├── utils.py
├── README.md
├── README_zh-CN.md
├── data/
│   ├── raw/
│   │   ├── drugbank.tab
│   │   └── twosides.csv
│   └── processed/
├── checkpoints/
└── results/
```

> `data_pre.py` must keep this filename because serialized PyG objects use `CustomData` from that module.

---

## Installation

The project requires compatible versions of:

- Python
- PyTorch
- PyTorch Geometric
- DGL
- RDKit
- NumPy
- pandas
- SciPy
- scikit-learn
- tqdm

Install PyTorch, PyTorch Geometric, and DGL according to the local CUDA or CPU environment. Then install the remaining dependencies:

```bash
pip install numpy pandas scipy scikit-learn tqdm rdkit
```

For an exact reproducibility record, export the final environment after the project runs successfully:

```bash
pip freeze > requirements-lock.txt
```

---

## Data format

Place the raw files under:

```text
data/raw/drugbank.tab
data/raw/twosides.csv
```

### DrugBank default columns

| Field | Column |
|---|---|
| Head drug ID | `ID1` |
| Tail drug ID | `ID2` |
| Head SMILES | `X1` |
| Tail SMILES | `X2` |
| Relation label | `Y` |

The default separator is a tab.

### TwoSIDES default columns

| Field | Column |
|---|---|
| Head drug ID | `Drug1_ID` |
| Tail drug ID | `Drug2_ID` |
| Head SMILES | `Drug1` |
| Tail SMILES | `Drug2` |
| Relation label | `New Y` |

The default separator is a comma.

Custom columns can be specified with:

```text
--head-id-col
--tail-id-col
--head-smiles-col
--tail-smiles-col
--relation-col
--delimiter
```

---

## Quick start

### 1. Build molecular graphs and similarity profiles

DrugBank:

```bash
python data_pre.py \
  --dataset drugbank \
  --raw-file ./data/raw/drugbank.tab \
  --output-root ./data/processed \
  --operation drug_data
```

TwoSIDES:

```bash
python data_pre.py \
  --dataset twosides \
  --raw-file ./data/raw/twosides.csv \
  --output-root ./data/processed \
  --operation drug_data
```

This step only needs to be run once per dataset.

### 2. Generate five split seeds

DrugBank:

```bash
for seed in 0 1 2 3 4; do
  python data_pre.py \
    --dataset drugbank \
    --raw-file ./data/raw/drugbank.tab \
    --output-root ./data/processed \
    --operation split \
    --mode both \
    --seed "$seed"
done
```

TwoSIDES:

```bash
for seed in 0 1 2 3 4; do
  python data_pre.py \
    --dataset twosides \
    --raw-file ./data/raw/twosides.csv \
    --output-root ./data/processed \
    --operation split \
    --mode both \
    --seed "$seed"
done
```

Use one of the following to generate a single protocol:

```text
--mode transductive
--mode inductive
```

### 3. Train and evaluate

Five transductive runs:

```bash
python train.py \
  --dataset drugbank \
  --mode transductive \
  --seeds 0 1 2 3 4 \
  --data-root ./data/processed \
  --result-root ./results \
  --checkpoint-root ./checkpoints
```

Five inductive runs:

```bash
python train.py \
  --dataset drugbank \
  --mode inductive \
  --seeds 0 1 2 3 4 \
  --data-root ./data/processed \
  --result-root ./results \
  --checkpoint-root ./checkpoints
```

Replace `drugbank` with `twosides` for TwoSIDES.

### Single-seed debugging

Strict paper mode requires exactly five seeds. For a one-seed debugging run:

```bash
python train.py \
  --dataset drugbank \
  --mode transductive \
  --seeds 0 \
  --no-strict-paper-hparams
```

Single-seed results should not be mixed with the reported five-run results.

---




## Model interface

The training script constructs the model with:

```python
model = gnn_model("GraphTransformer", net_params)
```

The model must accept dynamic values for:

```text
num_relations
sim_dim
similarity_dim
n_iter
L
n_heads
hidden_dim
num_atom_type
num_bond_type
```

The expected forward call is:

```python
logits = model(
    head_pyg,
    tail_pyg,
    head_dgl,
    tail_dgl,
    head_dgl.edata["feat"],
    tail_dgl.edata["feat"],
    relation,
    head_pyg.sim,
    tail_pyg.sim,
)
```

The output must contain one logit per DDI instance.

---


