# High-Energy Muon Calorimeter Regression Study

Public version of code used in "Calorimetric Measurement of Multi-TeV Muons via Deep Regression" by Jan Kieseler, Giles C. Strong, Filippo Chiandotto, Tommaso Dorigo, & Lukas Layer, (2021), [arXiv:2107.02119 [physics.ins-det]](https://arxiv.org/abs/2107.02119)

**Warning: This repo is currently being populated with a public version of the research code-base. It is currently incomplete, and has not yet been fully tested. Please check back soon**

## Installation

Please clone using:

```
git clone https://github.com/GilesStrong/calo_muon_regression.git
```

Self-hosted documentation, installation guide, and user guide are available by opening `./docs/build/html/index.html` with a web-browser.

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
