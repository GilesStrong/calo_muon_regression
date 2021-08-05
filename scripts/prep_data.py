import sys
from typer import Typer, Argument, Option
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import json
import os
from fastprogress import progress_bar
from tqdm import tqdm
from collections import defaultdict

from lumin.data_processing.file_proc import fold2foldfile, add_meta_data

sys.path.append('../')
from muon_regression.data_proc import get_dataset, Detector, calc_hl_feats, get_coord_grid, grid2matrix  # noqa E402

app = Typer()
rechit_x,rechit_y,rechit_z = None,None,None
hl_names = [
    "eabove",  # total energy recorded in the calorimeter in towers above the 0.1 GeV energy threshold (V[0] from the paper); - 0 
    "rmom",  # the missing transverse energy deposition in the xy plane (V[2] from the paper); - 1 
    "rmom1",  # same as (1) but uses only towers exceeding the 0.1 GeV threshold (V[3] from the paper); - 2 
    "r2m",  # the second moment of the energy distribution around the muon direction in the transverse plane (V[4] from the paper); - 3 
    "r2m1",  # same as (4) but computed only with the towers located in the first 400 mm-thick longitudinal section of the detector along z (V[5] from the paper); - 4 
    "r2m2",  # same as (4) but computed only with the towers located in the 400 < z < 800 mm region (V[6] from the paper); - 5 
    "r2m3",  # same as (4) but computed only with the towers located in the 800 < z < 1200 mm region (V[7] from the paper); - 6 
    "r2m4",  # same as (4) but computed only with the towers located in the 1200 < z < 1600 mm region (V[8] from the paper); - 7 
    "r2m5",  # same as (4) but computed only with the towers located in the z >= 1600 mm region (V[9] from the paper); - 8 
    "nclu",  # the number of muon trajectory-seeded clusters (V[10] from the paper); - 9 
    "ncellmax",  # the maximum number of cells among those clusters (V[12] from the paper); - 10
    "eclumax",  # the maximum total energy among those clusters (V[11] from the paper); - 11
    "maxdepthx",  # the maximum cluster extension in the x plane among the trajectory-seeded clusters; - 12
    "maxdepthy",  # the maximum cluster extension in the y plane among the trajectory-seeded clusters; - 13
    "maxdepthz",  # the maximum cluster extension in the z plane among the trajectory-seeded clusters; - 14
    "epercell",  # ratio between eclumax and ncellmax considering only the trajectory-seeded clusters; - 15
    "avgcells",  # average number of cells per cluster consideering only the trajectory-seeded clusters; - 16
    "nclu1",  # the number of clusters not seeded by a tower along the muon track (V[13] from the paper); - 17
    "ncellmax1",  # the maximum number of cells among those clusters (V[14] from the paper); - 18
    "eclumax1",  # the maximum total energy among those clusters (V[15] from the paper); - 19
    "epercell1",  # ratio between eclumax and ncellmax considering only clusters not seeded by a tower along the muon track; - 20
    "avgcells_off_trk",  # average number of cells per cluster considering only clusters not seeded by a tower along the muon track; - 21
    "ymom",  # E_x + E_x1 (for the computaion of E_x and E_x1 see V[2] from the paper) - 22
    "xmom",  # E_y + E_y1 (for the computaion of E_y and E_y1 see V[2] from the paper) - 23
    "emeas",  # measured energy a curvature fit; - 24
    "maxsum",  # the maximum energy of 3x3 clusters not aligned with muon direction; - 25
    "eb1",  # total energy in the towers below the 0.01 GeV threshold; - 26
    "ea1",  # total energy in the towers between the 0.01 GeV and 0.1 GeV thresholds; - 27
    "true_energy",  # true energy; - 28
]


def preproc_df(df:pd.DataFrame) -> None:
    df.emeas = np.clip(np.abs(df.emeas), 0, 10000)
    df.true_energy = df.true_energy.abs()
    # df = df[(df.gen_target >= 100) & (df.gen_target <= 8000)]


@app.command()
def singleroot_to_hdf5(in_file:  str=Argument(...,  help='Name of input ROOT file.'),
                       out_file: str=Argument(...,  help='Name of output hdf5 file.'),
                       tree_name:str=Argument('B4', help='Name of tree in which data is stored.'),
                       overwrite:bool=Option(False, help='Whether to overwrite exisitig outputs.')) -> None:
    r'''
    Example: python prep_data.py singleroot-to-hdf5 /eos/user/d/dorigo/Muon_E_loss/sample_1/1001.root .1001.hdf5
    '''

    if Path(out_file).exists() and not overwrite:
        print(f'{out_file} exists, skipping')
        return None

    df = get_dataset(in_file, tree_name=tree_name, as_flat=False, inputs=['rechit_energy'])

    detector = Detector(Path(in_file).parts[-2])
    if detector.shape is None: detector = Detector(Path(in_file).parts[-1])
    assert detector.shape is not None, f"Foldername {Path(in_file).parts[-2]} not recognised by Detector"

    e = np.vstack(df['rechit_energy'].values).reshape((-1,1,*detector.shape))
    y = df['true_energy'].values

    meta = {'e_mean':e.mean(),'e_sum':e.sum(), 'e_square_sum':np.square(e).sum(),'len':e.shape[0]}

    out_file = h5py.File(out_file, "w")
    out_file.create_dataset('E', shape=e.shape, dtype='float32', data=e.astype('float32'), compression="lzf")
    out_file.create_dataset('y', shape=y.shape, dtype='float32', data=y.astype('float32'), compression="lzf")
    out_file.create_dataset('meta', data=json.dumps(meta))
    out_file.close()


@app.command()
def allroot_to_hdf5(in_path:  str=Argument('/eos/user/d/dorigo/Muon_E_loss/sample_1', help='Path to raw ROOT files.'),
                    out_path: str=Argument('.',                                       help='Path to output directory.'),
                    tree_name:str=Argument('B4',                                      help='Name of tree in which data is stored.'),
                    n_files:int=Option(-1, help='Number of files to process. Negative argument to process all.')) -> None:
    r'''
    Example: python prep_data.py allroot-to-hdf5
    '''

    in_path,out_path = Path(in_path),Path(out_path)
    with open(in_path/'use_files.txt') as fin: files = list(sorted(fin.read().split('\n')[:-1]))
    if n_files >= 0: files = files[:n_files]
    for i, f in enumerate(tqdm(files)):
        f = Path(f)
        try:                   singleroot_to_hdf5(in_file=Path(f), out_file=out_path/f'{f.stem}.hdf5', tree_name=tree_name, overwrite=False)
        except Exception as e: print(e); pass


@app.command()
def get_csv_meta_data(csv_path:str=Argument(...,  help='Path to CSV files with HL features.')) -> None:
    csv_path = Path(csv_path)
    meta = defaultdict(float)
    for f in tqdm(csv_path.glob('*.asc')):
        df = pd.read_csv(f, sep=' ', header=None, names=hl_names)
        preproc_df(df)
        for c in hl_names:
            meta[f'hl_{c}_sum']         += df[c].sum()
            meta[f'hl_{c}_squared_sum'] += (df[c]**2).sum()
        meta['n_mu'] += len(df)
        if len(df[df.maxdepthx > 32]): print('x', f, df.loc[df.maxdepthz > 32, 'maxdepthx'].values)
        if len(df[df.maxdepthy > 32]): print('y', f, df.loc[df.maxdepthz > 32, 'maxdepthy'].values)
        if len(df[df.maxdepthz > 50]): print('z', f, df.loc[df.maxdepthz > 50, 'maxdepthz'].values)
    for c in hl_names:
        meta[f'hl_{c}_mean'] =  meta[f'hl_{c}_sum']/meta['n_mu']
        meta[f'hl_{c}_std']  = np.sqrt((meta[f'hl_{c}_squared_sum']/meta['n_mu'])-(meta[f'hl_{c}_mean']**2))
        if meta[f'hl_{c}_sum'] == 0: print(f'{c} has zero sum')
    print('Meta data is:', meta)
    with open(csv_path/'meta.json', 'w') as fout: json.dump(meta, fout)


@app.command()
def allhdf5_to_foldfile(in_path:      str=Argument(..., help='Path to HDF5 files produced by allroot_to_hdf5.'),
                        out_name:     str=Argument(..., help='Name of output file (excluding file suffix).'),
                        csv_path:     str=Argument('',  help='Path to CSV files with HL features. Leave blank to not include.'),
                        detector_name:str=Argument('sample_1', help='Name of detector as hardcoded in muon_regression.data_proc.detector.Detector.py'),
                        n_folds:             int=Option(19,    help='Number of folds to split the data into.', min=1),
                        frac_use:          float=Option(1.0,   help='Fraction of total data to process.', min=0, max=1),
                        mean_subtract_input:bool=Option(False, help='Whether to subtract the mean energy from the inputs.'),
                        hl_only:            bool=Option(False, help='Whether to only include the high-level features'),
                        esum_only:          bool=Option(False, help='Whether to only include the esum hl feature'),
                        tracker:            bool=Option(False, help='If true, will convert hits to Boolean'),
                        start_id:            int=Option(0,     help='Starting file number'),
                        include_target:     bool=Option(True,  help='Whether to include the true energy')):
    r'''
    Example: python prep_data.py allhdf5-to-foldfile ../data/skimmed/ sample_1 --frac-use=0.1 --n_folds=10
    '''
    import sparse

    inc_hl = csv_path != ''
    in_path,csv_path = Path(in_path),Path(csv_path)
    detector = Detector(detector_name)

    meta = None
    if inc_hl:
        with open(csv_path/'meta.json', 'r') as fin:
            m = json.load(fin)
            meta = m if meta is None else {**meta,**m}
    else:
        print('Computing meta data')
        e_sum,e2_sum,n_mu = 0,0,0
        for f in progress_bar([f for f in in_path.glob('*.hdf5')]):
            try:
                with h5py.File(f, 'r') as h5:
                    try:
                        tmp = json.loads(h5['meta'][()])
                        e_sum  += tmp['e_sum']
                        e2_sum += tmp['e_square_sum']
                        n_mu   += tmp['len']
                    except KeyError:
                        continue
            except OSError:
                continue
        meta = {'e_mean': e_sum/(n_mu*np.prod(detector.shape)), 'n_mu': n_mu}
        meta['e_std'] = np.sqrt((e2_sum/(n_mu*np.prod(detector.shape)))-(meta['e_mean']**2))
        print(f'Meta data is: {meta}')

    fold_sz = int(frac_use*meta['n_mu']/n_folds)
    print(f'Processing {n_folds} folds of {fold_sz} muons')
    os.system(f'rm {out_name}.hdf5')
    if '/' in out_name: os.makedirs(out_name[:out_name.rfind('/')], exist_ok=True)
    out_file = h5py.File(f'{out_name}.hdf5', "w")

    hits,hl_x,targs,fold_idx = None,None,None,0
    files = list(sorted([f for f in in_path.glob('*.hdf5')]))[start_id:]

    for i,f_idx in enumerate(progress_bar(files)):
        print('Reading file', f_idx)
        try:            h5 = h5py.File(f_idx, 'r')
        except OSError: continue
        try:
            # Load hits
            if not hl_only:
                e = sparse.as_coo(h5['E'][()])
                if mean_subtract_input: e -= meta['e_mean']
                if hits is None: hits = e
                else:            hits = sparse.concatenate((hits,e),0)

            # Load targets
            y = np.abs(h5['y'][()])
            if targs is None: targs = y
            else:             targs = np.concatenate((targs,y),0)

            # Load HL feats
            if inc_hl:
                df = pd.read_csv(csv_path/f'{f_idx.stem}.asc', sep=' ', header=None, names=hl_names)
                preproc_df(df)
                assert len(df) == len(y), f'Length mismatch in {f_idx}: HDF5 contains {len(y)} muons, but csv contains {len(df)}'
                d = np.abs(df.true_energy-targs[-len(df):]).mean()
                assert d < 1e-2, f"Target energies do not match during reading in {f_idx}, by an average of {d}"
                if esum_only: df.drop(columns=[f for f in df.columns if f not in ['eabove','ebelow', 'true_energy']], inplace=True)
                for f in [c for c in df.columns if c != 'true_energy']:
                    df[f] -= meta[f'hl_{f}_mean']
                    df[f] /= meta[f'hl_{f}_std']
                assert not df.isnull().values.any(), 'NaNs found in DataFrame during reading'
                if hl_x is None: hl_x = df
                else:            hl_x = hl_x.append(df, ignore_index=True)

        except KeyError:
            continue
        h5.close()

        while len(targs) >= fold_sz:
            df = pd.DataFrame(data={'target':targs[:fold_sz]})
            if hl_x is not None:
                df = df.join(hl_x[:fold_sz])
                d = np.abs(df.true_energy-df.target).mean()
                assert d < 1e-2, f"Target energies do not match during saving, by an average of {d}"
                assert not df.isnull().values.any(), 'NaNs found in DataFrame during saving'
                df.drop(columns=['true_energy'], inplace=True)
            print(f'Saving fold {fold_idx}, containing {len(df)} muons')
            if not include_target: df.target = -1

            fold2foldfile(df, out_file,
                          tensor_data=None if hl_only else np.vstack((hits[:fold_sz].data if not tracker else np.ones_like(hits[:fold_sz].data),
                                                                      hits[:fold_sz].coords)).astype('float32'),
                          fold_idx=fold_idx,
                          cont_feats=[] if hl_x is None else [f for f in df.columns if f != 'target'],
                          cat_feats=[],
                          targ_feats='target',
                          targ_type='float32',
                          compression='lzf')

            if not hl_only: hits = hits[fold_sz:]
            targs = targs[fold_sz:]
            if hl_x is not None: hl_x = hl_x[fold_sz:].reset_index(drop=True)
            fold_idx += 1
            if fold_idx == n_folds: break
        if fold_idx == n_folds: break

    add_meta_data(out_file=out_file,
                  feats=df.columns,
                  cont_feats=[f for f in df.columns if f != 'target'],
                  cat_feats=[],
                  cat_maps=None,
                  targ_feats=None  if hl_only else 'true_energy',
                  tensor_name=None if hl_only else 'rechit_energy',
                  tensor_shp=None  if hl_only else  hits.shape[1:],
                  tensor_is_sparse=True)


@app.command()
def allhdf5_to_foldfile_asmatrix(in_path:      str=Argument(..., help='Path to HDF5 files produced by allroot_to_hdf5.'),
                                 out_name:     str=Argument(..., help='Name of output file (excluding file suffix).'),
                                 csv_path:     str=Argument('',  help='Path to CSV files with HL features. Leave blank to not include.'),
                                 detector_name:str=Argument('sample_1', help='Name of detector as hardcoded in muon_regression.data_proc.detector.Detector.py'),
                                 n_folds:             int=Option(19,    help='Number of folds to split the data into.', min=2),
                                 frac_use:          float=Option(1.0,   help='Fraction of total data to process.', min=0, max=1),
                                 mean_subtract_input:bool=Option(False, help='Whether to subtract the mean energy from the inputs.'),
                                 n_max:               int=Option(5000,  help="Number of hits to include, ordered by decreasing energy")) -> None:
    r'''
    Example: python prep_data.py allhdf5-to-foldfile-asmatrix ../data/skimmed/ sample_1 --frac-use=0.1 --n_folds=10
    TODO: Update this to work with new HL features
    '''

    import sparse

    inc_hl = csv_path != ''
    in_path,csv_path = Path(in_path),Path(csv_path)
    detector = Detector(detector_name)
    coords = get_coord_grid(detector.shape)

    meta = None
    if inc_hl:
        with open(csv_path/'meta.json', 'r') as fin:
            m = json.load(fin)
            meta = m if meta is None else {**meta,**m}
    else:
        print('Computing meta data')
        e_sum,e2_sum,n_mu = 0,0,0
        for f in progress_bar([f for f in in_path.glob('*.hdf5')]):
            try:
                with h5py.File(f, 'r') as h5:
                    try:
                        tmp = json.loads(h5['meta'][()])
                        e_sum  += tmp['e_sum']
                        e2_sum += tmp['e_square_sum']
                        n_mu   += tmp['len']
                    except KeyError:
                        continue
            except OSError:
                continue
        meta = {'e_mean': e_sum/(n_mu*np.prod(detector.shape)), 'n_mu': n_mu}
        meta['e_std'] = np.sqrt((e2_sum/(n_mu*np.prod(detector.shape)))-(meta['e_mean']**2))
        print(f'Meta data is: {meta}')

    fold_sz = int(frac_use*meta['n_mu']/n_folds)
    print(f'Processing {n_folds} folds of {fold_sz} muons')
    os.system(f'rm {out_name}.hdf5')
    if '/' in out_name: os.makedirs(out_name[:out_name.rfind('/')], exist_ok=True)
    out_file = h5py.File(f'{out_name}.hdf5', "w")

    hits,hl_x,targs,fold_idx = None,None,None,0
    files = list(sorted([f for f in in_path.glob('*.hdf5')]))
    for f_idx in progress_bar(files):
        try:            h5 = h5py.File(f_idx, 'r')
        except OSError: continue
        try:
            # Load hits
            e = sparse.as_coo(h5['E'][()])
            if mean_subtract_input: e -= meta['e_mean']
            if hits is None: hits = e
            else:            hits = sparse.concatenate((hits,e),0)

            # Load targets
            y = np.abs(h5['y'][()])
            if targs is None: targs = y
            else:             targs = np.concatenate((targs,y),0)

            # Load HL feats
            if inc_hl:
                df = pd.read_csv(csv_path/f'{f_idx.stem}.asc', sep=' ', header=None, names=hl_names)
                preproc_df(df)
                assert len(df) == len(y), f'Length mismatch in {f_idx}: HDF5 contains {len(y)} muons, but csv contains {len(df)}'
                d = np.abs(df.true_energy-targs[-len(df):]).mean()
                assert d < 1e-2, f"Target energies do not match during reading in {f_idx}, by an average of {d}"
                for f in [c for c in df.columns if c != 'true_energy']:
                    df[f] -= meta[f'hl_{f}_mean']
                    df[f] /= meta[f'hl_{f}_std']
                assert not df.isnull().values.any(), 'NaNs found in DataFrame during reading'
                if hl_x is None: hl_x = df
                else:            hl_x = hl_x.append(df, ignore_index=True)

        except KeyError:
            continue
        h5.close()

        while len(targs) >= fold_sz:
            df = pd.DataFrame(data={'target':targs[:fold_sz]})
            if hl_x is not None:
                df = df.join(hl_x[:fold_sz])
                d = np.abs(df.true_energy-df.target).mean()
                assert d < 1e-2, f"Target energies do not match during saving, by an average of {d}"
                assert not df.isnull().values.any(), 'NaNs found in DataFrame during saving'
                df.drop(columns=['true_energy'], inplace=True)
            print(f'Saving fold {fold_idx}, containing {len(df)} muons')

            fold2foldfile(df, out_file,
                          tensor_data=grid2matrix(hits[:fold_sz].todense(), coords, n_max, row_wise=True).astype('float32'),
                          fold_idx=fold_idx,
                          cont_feats=[] if hl_x is None else [f for f in df.columns if f != 'target'],
                          cat_feats=[],
                          targ_feats='target',
                          targ_type='float32',
                          compression='lzf')
            hits = hits[fold_sz:]
            targs = targs[fold_sz:]
            if hl_x is not None: hl_x = hl_x[fold_sz:].reset_index(drop=True)
            fold_idx += 1
            if fold_idx == n_folds: break
        if fold_idx == n_folds: break

    add_meta_data(out_file=out_file,
                  feats=df.columns,
                  cont_feats=[f for f in df.columns if f != 'target'],
                  cat_feats=[],
                  cat_maps=None,
                  targ_feats='true_energy',
                  tensor_name='rechit_energy',
                  tensor_shp=(n_max,4),
                  tensor_is_sparse=False)


if __name__ == "__main__": app()
