# SSE-DDI

SSE-DDI is a molecular structure-driven framework for drug–drug interaction (DDI) prediction. The model learns drug representations from SMILES-derived molecular graphs and Morgan-fingerprint-based structural similarity profiles. Its core module, Substructure-Selective Encoding (SSE), performs bond-level representation learning on molecular line graphs to selectively encode DDI-informative local substructures.

## Requirements

Please install the required Python packages before running the code.

```bash
python >= 3.8
torch
torch-geometric
rdkit
numpy
pandas
scikit-learn
```

## Dataset

This project supports the DrugBank and TwoSIDES datasets.

### DrugBank

The DrugBank dataset can be obtained from the official DrugBank website:

```text
https://go.drugbank.com/
```

Users may need to register an account and follow the DrugBank data access or licensing requirements.

### TwoSIDES

The TwoSIDES dataset can be obtained from the Tatonetti Lab / nSIDES resources:

```text
https://tatonettilab.org/resources/tatonetti-stm.html
https://tatonettilab.org/offsides/
https://github.com/tatonetti-lab/nsides-release/releases
```

TwoSIDES provides drug–drug–side-effect associations for polypharmacy side effects.

After downloading the datasets, place the raw data files in the corresponding dataset directory. An example directory structure is shown below:

```bash
data/
├── drugbank/
│   └── raw files
└── twosides/
    └── raw files
```

Please make sure that the dataset path and file names are consistent with the settings used in `data_pre.py`.

## Data Preparation

Before training the model, run `data_pre.py` to generate the three-fold data splits.

For the DrugBank dataset:

```bash
python data_pre.py -d drugbank -o all
```

For the TwoSIDES dataset:

```bash
python data_pre.py -d twosides -o all
```

Arguments:

- `-d`: dataset name, such as `drugbank` or `twosides`
- `-o`: output mode, where `all` generates all required processed data files

## Training

After data preprocessing, run `train.py` to train SSE-DDI:

```bash
python train.py
```

The training script will load the processed dataset and train the SSE-DDI model using the default configuration.

## Running Pipeline

For DrugBank:

```bash
# Step 1: Generate five-fold data
python data_pre.py -d drugbank -o all

# Step 2: Train SSE-DDI
python train.py
```

For TwoSIDES:

```bash
# Step 1: Generate five-fold data
python data_pre.py -d twosides -o all

# Step 2: Train SSE-DDI
python train.py
```
