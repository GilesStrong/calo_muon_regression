# User Guide

## Data processing

Data processing is performed in several stages, starting from the individual ROOT files. Processing functions are primarily found in `muon_regression/scripts/prep_data.py` and can be called as CLI apps via [typer](https://typer.tiangolo.com/) (call with `--help` to see options and arguments).

### ROOT -> HDF5 Hits

`singleroot-to-hdf5` acts on a single ROOT file, e.g. `python prep_data.py singleroot-to-hdf5 /eos/user/d/dorigo/Muon_E_loss/sample_1/1001.root .1001.hdf5`, and saves the reco hit deposits, true energy, and a few HL features to lzf-compressed HDF5 files, via h5py.

- Hits are stored in 'E' as a dense 4D tensor with dimensions (muon, z-cell, x-cell, y-cell).
- The true energy is stored in 'y' as a vector.
- Hl feats are stored in 'hl_x' as a matrix with dimensions (muon, HL feats). These HL feats are now obsolete, since computation of HL feats is now performed in c++, as discussed later.
- Meta data about the muons, and the mean and standard deviation of the HL feats is stored in 'meta'.

Some data files are corrupted. `muon_regression/scripts/file_check.py` can be run to loop through all ROOT files in a specified directory and produce use_files.txt, which lists all the valid files.

`allroot-to-hdf5` can be then used to loop over all ROOT files listed in use_files.txt and runs `singleroot-to-hdf5` on each of them.

### ROOT -> CSV HL Feats

Computation of HL feats is now performed in c++, due to loops in the initial implementation of the clustering algorithm, using `muon_regression/macros/*/ReadMuELossTree*.*`. This must be run as a ROOT macro over each file listed in `use_files.txt`, and produces CSV files of the HL feats. There two version:

- `4TeV/ReadMuELossTree5.*` is the latest version.
- `2TeV/ReadMuELossTree3.*` is the version used for the August preprint.

### HDF5 + CSV -> LUMIN foldfile (HDF5)

`allhdf5-to-foldfile` is used to combine the hits and true-energy files with the CSV HL feats, and has a variety of options. The output is a single lzf-compressed HDF5 file suitable for direct usage in LUMIN. Hits are stored as sparse arrays. Muons are pre-split randomly into a specified number of folds.
