# Installation Guide

## Requirements

- Python >= 3.6
- Linux or Mac OSX

## Installation

The repository can be cloned using:
```
git clone git@github.com:GilesStrong/muon_regression.git
```

Providing one uses Conda for one's Python environments, a suitable Python environment can be built using:
```
conda env create -f environment.yml
```

This will create a new Conda environment called `muon-regression`, which can be activated using:
```
conda activate muon-regression
```

The bleeding edge version of [LUMIN](https://github.com/GilesStrong/lumin) is recommended. This can be installed using:
```
git clone https://github.com/GilesStrong/lumin.git && cd lumin && git checkout verbose_matrix && pip install -e . && cd ..
```
The `-e` flag links the lumin dir to pip so that local changes are immediately available when running the code: one can run `git pull` in `./lumin` to get the latest version available for use.

Full use of LUMN requires the bleeding-edge version of [pdplot](https://github.com/SauceCat/PDPbox), which is available by running:

```
git clone https://github.com/SauceCat/PDPbox.git && cd PDPbox && pip install -e . && cd ..
```
