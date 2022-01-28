[![DOI](https://zenodo.org/badge/392616618.svg)](https://zenodo.org/badge/latestdoi/392616618)

# High-Energy Muon Calorimeter Regression Study

Public version of code used in "Calorimetric Measurement of Multi-TeV Muons via Deep Regression" by Jan Kieseler, Giles C. Strong, Filippo Chiandotto, Tommaso Dorigo, & Lukas Layer, [The European Physical Journal C volume 82, Article number: 79 (2022)](https://link.springer.com/article/10.1140%2Fepjc%2Fs10052-022-09993-5)

**Warning: This repo is currently being populated with a public version of the research code-base. It is currently incomplete, and has not yet been fully tested. Please check back soon**

## Installation

Please clone using:

```
git clone https://github.com/GilesStrong/calo_muon_regression.git
```

Providing one uses Conda for one's Python environments, a suitable Python environment can be built using:
```
conda env create -f environment.yml
```

This will create a new Conda environment called `muon-regression`, which can be activated using:
```
conda activate calo_muon_regression
```

Alternatively:
```
pip install -r requirements.txt
```
may be used to install the dependencies.

Self-hosted documentation, installation guide, and user guide are available by opening `./docs/build/html/index.html` with a web-browser.

For GPU use (which is highly recommended), please separately install a suitable version of PyTorch based on your drivers.

## Data

Preprocessed datasets are available from https://doi.org/10.5281/zenodo.5163817. The example commands assume the data files are stored in `./data`
Raw data may be made available at a later date, since exceeds the size limits for Zenodo.

## Training

Model training and evaluation can be accomplished by using the script in `./scripts/train_stride2_model.py`, which takes a variety of options to produce different models. As an example, the following line will train the final model used in out paper:

```
cd scripts
python train_stride2_model.py train --gpu-id 0 --n-models 5 --extra-val-data  cnn_hl_full_ensemble ../data/feb_calo_36_hl.hdf5
```

The results and model will be saved in `./results`.

Alternatively, models can be trained within Jupyter Notebooks.

## Authors

- Jan Kieseler
- Giles C. Strong
- Filippo Chiandotto
- Tommaso Dorigo
- Lukas Layer
