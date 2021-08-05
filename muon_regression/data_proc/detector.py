from typing import Union
from pathlib import Path
import numpy as np

__all__ = ['Detector']


class Detector():
    r'''
    Stores various information about the detector that was used for each data-sample.

    Arguments:
        fname: Filename from which data will be loaded. Used to look up hard-coded detector configurations.
    '''

    shape,interaction_length,x,y,x,grid = None,None,None,None,None,None

    def __init__(self, fname:Union[str,Path]):
        if not isinstance(fname, Path): fname = Path(fname)
        self.fname = fname
        self.configure_params(fname.stem)

    def __repr__(self) -> str:
        return f'''Detector corresponding to fname: {self.fname} \
                   \nWith (z,x,y) shape: {self.shape} \
                   \nAnd depth corresponding to {self.interaction_length} interaction lengths.'''

    def configure_params(self, fname:str) -> None:
        r'''
        Sets detector parameters based on the file name.
        The list of file names should continually be updated to ensure that parameters are correctly configured

        Arguments:
            fname: The stem of the filename from which data will be loaded
        '''

        self.shape,self.interaction_length = (50,32,32),10
        self.z,self.x,self.y = np.linspace(0,1991.36, 50),np.linspace(-58.125,58.125,32),np.linspace(-58.125,58.125,32)
        self.grid = self._make_grid()

    def _make_grid(self) -> np.ndarray:
        x,y,z = np.zeros(np.prod(self.shape)),np.zeros(np.prod(self.shape)),np.zeros(np.prod(self.shape))
        i = 0
        for a in self.z:
            for b in self.x:
                for c in self.y:
                    z[i],x[i],y[i] = a,b,c
                    i += 1
        x,y,z = x.reshape(1,*self.shape),y.reshape(1,*self.shape),z.reshape(1,*self.shape)
        return np.concatenate((z,x,y), axis=0)

    