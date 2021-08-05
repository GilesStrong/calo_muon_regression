from typing import Union, Optional, List, Tuple
from pathlib import Path
import pandas as pd
import uproot
import numpy as np

__all__ = ['get_dataset', 'calc_hl_feats', 'get_coord_grid', 'grid2matrix']


def get_dataset(fname:Union[str,Path], tree_name:str, as_flat:bool=False, start:Optional[float]=None, stop:Optional[float]=None,
                inputs:Optional[List[str]]=None, targets:Union[str,List[str]]='true_energy') -> pd.DataFrame:
    r'''
    Loads a `pandas.DataFrame` from the referenced filename.
    If `as_flat==True`, then a flat `DataFrame` is returned with columns per 'pixel' per input feature.
    Otherwise returns a `DataFrame` in which each input has a single column containing an array of all 'pixel' values.

    Arguments:
        fname: file name from which to load data. `.root` suffix added if missing.
        tree_name: tree in root file from which to load data
        as_flat: whether to return a flattened `DataFrame` or one with arrays in rows
        start: if set, will load  entries from this point
        stop: if set, will stop loading entries after this point
        inputs: if set, will load specified branches from the data, otherwise suitable default branches will be loaded
        targets: if set, will load the specified branches as target features, otherwise will load 'true_energy'

    Returns:
        `pandas.DataFrame` with inputs and targets
    '''

    if not isinstance(fname, Path): fname = Path(fname)
    if not isinstance(targets, list): targets = [targets]
    if fname.suffix == '': fname = fname.with_suffix('.root')
    if not fname.exists(): raise FileNotFoundError(fname)
    
    if inputs is None: inputs = ['rechit_energy']
    tree = uproot.open(fname)[tree_name]
    if as_flat:
        df = tree.pandas.df(branches=inputs, entrystart=start, entrystop=stop).unstack().reset_index()
        dft = tree.pandas.df(branches=targets, entrystart=start, entrystop=stop)
        df.drop(columns=['entry'], inplace=True)
        df.columns = [f'{c[0][:c[0].find("_")]}_{c[1]}_{c[0][c[0].find("_")+1:]}'for c in df.columns.values]
        assert len(df.index) == len(df.index), "Length mismatch between inputs and targets"
        df[dft.columns] = dft[dft.columns]
    else:
        df = tree.pandas.df(branches=targets+inputs, entrystart=start, entrystop=stop, flatten=False)
    return df


def calc_hl_feats(rec_e, rec_x, rec_y, rec_z, true_x, true_y) -> pd.Series:
    def _calc_sume2rz(rec_e, sume, dist2, rec_z, zlims=None):
        # Second momenta of energy distribution
        if sume == 0: return 0
        
        # Slice in z
        if zlims is not None: 
            mask = (rec_z > zlims[0]) & (rec_z < zlims[1])
            rec_e = rec_e[mask]
            dist2 = dist2[mask]
            rec_z = rec_z[mask]    
        
        sume2rz = np.sum(dist2*rec_e)
        return sume2rz / sume
    
    # Definition of bins for the energy towers
    bins = np.arange(0,2001, 400)
    bins = list(zip(bins[:-1], bins[1:]))
    
    xrel = rec_x-true_x
    yrel = rec_y-true_y
    
    """
    Transverse energy imbalance (computed as sqrt(sum(E*dx)^2+sum(E*dy)^2) ) where dx and dy are signed spatial
    differences between muon position in xy and cell center in xy, and E is the deposited energy in the cell.
    """
    sumex = np.sum(xrel*rec_e)
    sumey = np.sum(yrel*rec_e)
    sume = np.sum(rec_e)
    rmom = np.sqrt((sumex**2)+(sumey**2))/sume if sume != 0 else 0
    
    """
    Five values of the second moment of the energy distribution around the muon track, computed with the towers
    belonging to each of five 40-cm sections along the detector. These can be computed by computing the radial
    distance between a cell center and the muon position in xy, and extracting, 
    for each of the five z sections of the detector,
    the value sum(DR^2*E)/sum(E), where sums run on all cells, DR is the radial distance, and E is the cell energy. 
    """
    dist2 = (xrel**2)*(yrel**2)
    hl_feat = pd.Series({f'sume2rz_{i}':_calc_sume2rz(rec_e, sume, dist2, rec_z, zlims=b) for i,b in enumerate(bins)})
    
    # Add the transverse imbalance
    hl_feat['rmom'] = rmom
    hl_feat['esum'] = sume

    return hl_feat


def get_coord_grid(shp:Tuple[int,int,int]) -> np.ndarray:
    n = np.prod(shp)
    x,y,z = np.zeros(n),np.zeros(n),np.zeros(n)
    i = 0
    for a in np.linspace(-1.,1.,shp[0]):
        for b in np.linspace(-1.,1.,shp[1]):
            for c in np.linspace(-1.,1.,shp[2]):
                z[i],x[i],y[i] = a,b,c
                i += 1
    x,y,z = x.reshape(1,*shp),y.reshape(1,*shp),z.reshape(1,*shp)
    return np.concatenate((z,x,y), axis=0).reshape(3,n)[None,:]


def grid2matrix(rec_e:np.ndarray, coords:Tuple[int,int,int], n_max:int, row_wise:bool) -> np.ndarray:
    # Append coords
    g = np.concatenate((rec_e if len(rec_e.shape) == 3 else rec_e.reshape((rec_e.shape[0], 1, -1)),
                        np.broadcast_to(coords,(rec_e.shape[0],*coords.shape[1:]))),axis=1)
    
    # Order by energy and cut
    args = g[:,0].argsort()[:,::-1][:,None,:]
    i = np.arange(g.shape[0])[:,None][:,:,None]
    j = np.arange(g.shape[1])[None,:][:,:,None]
    g = g[i,j,args][:,:,:n_max]
    
    return g if not row_wise else g.swapaxes(1,2)
