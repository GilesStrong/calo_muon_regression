from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
from typing import List, Optional
from abc import ABC, abstractmethod

from lumin.utils.statistics import bootstrap_stats
from lumin.utils.multiprocessing import mp_run

from ..fqs import single_quartic


__all__ = ['LinFunc', 'QuadFunc', 'LogFunc', 'PowerFunc', 'QuartFunc', 'fit_pred']


class AbsFunc(ABC):
    def __init__(self, n_params:int, max_x:Optional[float]=8000):
        self.n_params,self.max_x,self.params = n_params,max_x,None

    def is_fitted(self) -> None: return self.params is not None

    def set_params(self,params:List[float]) -> None:
        self.params = params
        if self.max_x is not None:
            self.cut = self._func(self.max_x, self.params)
            self.grad = self._grad(self.cut)

    def inv_func(self, x:np.ndarray) -> np.ndarray:
        corr = self._inv(x)
        if self.max_x is not None: corr[x > self.cut] = self.max_x+((x[x > self.cut]-self.cut)*self.grad)
        return corr

    def func(self, x:np.ndarray, *args:float) -> np.ndarray: return self._func(x,self.params if self.is_fitted() else [*args])

    def fit(self, x:np.ndarray, y:np.ndarray, sigma:np.ndarray) -> None:
        p, _ = curve_fit(self.func, x, y, p0=[1 for _ in range(self.n_params)], sigma=sigma)
        self.set_params(p)

    @abstractmethod
    def _func(self, x:np.ndarray, params:Optional[List[float]]=None) -> np.ndarray: pass
    
    @abstractmethod
    def _inv(self, x:np.ndarray) -> np.ndarray: pass

    @abstractmethod
    def _grad(self, x:float) -> float: pass

    def string(self) -> str: return ' '.join([f'{p:.2f}' for p in self.params])


class LinFunc(AbsFunc):
    def __init__(self, *args, **kwargs): super().__init__(n_params=2, max_x=None)

    def _func(self, x:np.ndarray, params:Optional[List[float]]=None) -> np.ndarray:
        a,b = params
        return (a*x)+b

    def _inv(self, x:np.ndarray) -> np.ndarray:
        a,b = self.params
        return (x-b)/a

    def _grad(self, x:float) -> float:
        a,b = self.params
        return x/a

    def string(self) -> str:
        a,b = self.params
        return fr'${a:.2f}' + r'E_{\mathrm{True}}' + fr'+ {b:.2f}$'


class QuadFunc(AbsFunc):
    def __init__(self, max_x:Optional[float]=8000): super().__init__(n_params=3, max_x=max_x)

    def _func(self, x:np.ndarray, params:Optional[List[float]]=None) -> np.ndarray:
        a,b,c = params
        return (a*(x**2))+(b*x)+c

    def _inv(self, x:np.ndarray) -> np.ndarray:
        a,b,c = self.params
        return (-b+np.sqrt((b**2)-(4*a*(c-x))))/(2*a)

    def _grad(self, x:float) -> float:
        a,b,c = self.params
        return 1/np.sqrt((4*a*(x-c))+(b**2))

    def string(self) -> str:
        a,b,c = self.params
        return fr'$({a:.2E})' + r'E_{\mathrm{True}}' + fr'^2 + {b:.2f}' + r'E_{\mathrm{True}}' + fr' + {c:.2f}$'


class LogFunc(AbsFunc):
    def __init__(self, max_x:Optional[float]=8000): super().__init__(n_params=3, max_x=max_x)

    def _func(self, x:np.ndarray, params:Optional[List[float]]=None) -> np.ndarray:
        a,b,c = params
        return (a*(np.log(x+b)))+c

    def _inv(self, x:np.ndarray) -> np.ndarray:
        a,b,c = self.params
        return np.exp((x-c)/a)-b

    def _grad(self, x:float) -> float:
        a,b,c = self.params
        return np.exp((x-c)/a)/a

    def string(self) -> str:
        a,b,c = self.params
        return fr'${a:.2f}\times\ln (' + r'E_{\mathrm{True}}' + fr' + {b:.2f}) + {c:.2f}$'


class PowerFunc(AbsFunc):
    def __init__(self, max_x:Optional[float]=8000): super().__init__(n_params=3, max_x=max_x)

    def _func(self, x:np.ndarray, params:Optional[List[float]]=None) -> np.ndarray:
        a,b,c = params
        return a*(x**b)+c

    def _inv(self, x:np.ndarray) -> np.ndarray:
        a,b,c = self.params
        return ((x-c)/a)**(1/b)

    def _grad(self, x:float) -> float:
        a,b,c = self.params
        return (((x-c)/a)**((1/b)-1))/(a*b)

    def string(self) -> str:
        a,b,c = self.params
        return fr'${a:.2f}' + r'E_{\mathrm{True}}' + '^{' + f'{b:.2f}'+'}' + fr'{c:.2f}$'


class QuartFunc(AbsFunc):
    def __init__(self, max_x:Optional[float]=8000): super().__init__(n_params=5, max_x=max_x)

    def _func(self, x:np.ndarray, params:Optional[List[float]]=None) -> np.ndarray:
        a,b,c,d,e = params
        return (a*(x**4))+(b*(x**3))+(c*(x**2))+(d*x)+e

    def _inv(self, x:np.ndarray) -> np.ndarray:
        r'''TODO: replace with quartic formula?'''
        a,b,c,d,e = self.params
        rs = []
        for v in x:
            try: rs.append(np.max([r.real for r in single_quartic(a,b,c,d,e-v) if r.imag == 0]))
            except ValueError: raise ValueError(f'pred {v} has no real roots: {single_quartic(a,b,c,d,e-v)}')
        return np.array(rs)

    def _grad(self, x:float) -> float:
        r'''TODO: replace with quartic formula?'''
        x0 = x-0.1
        a,b,c,d,e = self.params
        y0 = np.max([r.real for r in single_quartic(a,b,c,d,e-x0) if r.imag == 0])
        return (self.max_x-y0)/(x-x0)

    def string(self) -> str:
        a,b,c,d,e = self.params
        return fr'$({a:.2E})' + r'E_{\mathrm{True}}' + fr'^4 + ({b:.2E})' + r'E_{\mathrm{True}}' + fr'^3 + ({c:.2E})' + r'E_{\mathrm{True}}' + fr'2 + {d:.2f}' + r'E_{\mathrm{True}}' + fr' + {e:.2f}$'


def fit_pred(df:pd.DataFrame, fit_func:AbsFunc, centre:str='mean', bins:np.ndarray=np.linspace(100,8000,80), n_bs:int=10000) -> pd.DataFrame:
    args = [{'data':df.loc[(df.gen_target >= bins[i]) & (df.gen_target < bins[i+1]), 'pred'].values, 'name':f'bin_{i}', centre:True, 'n':n_bs} for i in range(len(bins[:-1]))]
    bs_stats = mp_run(args, bootstrap_stats)
    
    centres,uncs,targs,p16,p84 = [],[],[],[],[]
    for i in range(len(bins[:-1])): 
        targs.append(bins[i]+((bins[i+1]-bins[i])/2))
        centres.append(np.mean(bs_stats[f'bin_{i}_mean']) if centre == 'mean' else np.median(bs_stats[f'bin_{i}_median']))
        p16.append(np.percentile(bs_stats[f'bin_{i}_{centre}'], 15.865))
        p84.append(np.percentile(bs_stats[f'bin_{i}_{centre}'], 84.135))
        uncs.append(np.std(bs_stats[f'bin_{i}_{centre}']))
        
    agg = pd.DataFrame(data={'gen_target':targs, f'pred_{centre}':centres, 'pred_std':uncs, 'pred_p16':p16, 'pred_p84':p84})    
    fit_func.fit(agg.gen_target, agg[f'pred_{centre}'], sigma=agg['pred_std'])
    return agg
    