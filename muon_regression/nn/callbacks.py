from typing import Callable, Union
import numpy as np

from torch import Tensor
from torch.nn import functional as F
from lumin.nn.callbacks.data_callbacks import AbsWeightData

__all__ = ['WeightDataByTargetFunc', 'DownweightHighEMuons']


class WeightDataByTargetFunc(AbsWeightData):
    def __init__(self, func:Callable[[Union[np.ndarray,Tensor]], Union[np.ndarray,Tensor]], on_eval:bool):
        super().__init__(on_eval=on_eval)
        self.func = func
        
    def weight_func(self, y:Union[np.ndarray,Tensor], **kargs) -> Union[np.ndarray,Tensor]: return self.func(y)


class DownweightHighEMuons(AbsWeightData):
    def __init__(self):
        super().__init__(on_eval=True)

    def on_train_begin(self) -> None:
        max_e = self.model.fit_params.fy.get_column('targets', n_folds=1, fold_idx=0).max()
        if   max_e <= 4000: self.offset,self.div = 3000,[150]
        elif max_e <= 8000: self.offset,self.div = 5000,[300,600]  
        else: raise ValueError(f'DownweightHighEMuons not yet configured for data with energy range up to {max_e}, please hard code offset and divisor')
        
    def weight_func(self, y:Union[np.ndarray,Tensor], **kargs) -> Union[np.ndarray,Tensor]:
        w = (1-F.sigmoid((y-self.offset)/self.div[0])) if isinstance(y, Tensor) else (1-(1/(1+np.exp(-(y-self.offset)/self.div[0]))))
        if len(self.div) > 1:
            w[y > self.offset] = (1-F.sigmoid((y[y > self.offset]-self.offset)/self.div[1])) if isinstance(y, Tensor) \
                else (1-(1/(1+np.exp(-(y[y > self.offset]-self.offset)/self.div[1]))))
        return w
