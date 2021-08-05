from collections import defaultdict
import numpy as np
from typing import List, Dict

__all__ = ['preproc_data']


def preproc_data(arr:np.ndarray, channel_names:List[str], apply:bool=False) -> Dict[str,Dict[str,float]]:
    r'''
    Computed preprocessing values per channel (dimension 1 of array) and optionally applies preprocessing to array
    
    Arguments:
        arr: input data
        channel_names: names of each channel
        apply: if True will preprocess array inplace
        
    Returns:
        Dictionary mapping channel to mean and standard deviation of channel
    '''

    preproc = defaultdict(dict)
    for i, f in enumerate(channel_names):
        print()
        preproc[f]['mean'] = arr[:,i:i+1].mean()
        preproc[f]['std']  = arr[:,i:i+1].std()
        if apply:
            arr[:,i:i+1] -= preproc[f]['mean']
            arr[:,i:i+1] /= preproc[f]['std']
    return preproc
