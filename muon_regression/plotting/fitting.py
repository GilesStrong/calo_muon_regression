from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
from typing import Callable, List, Optional
from abc import ABC, abstractmethod

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
        return fr'${a:.2f}\times x + {b:.2f}$'


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
        return fr'$({a:.2E})\times x^2 + {b:.2f}\times x + {c:.2f}$'


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
        return fr'${a:.2f}\times\ln (x + {b:.2f}) + {c:.2f}$'


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
        return fr'${a:.2f}\times x^'+'{' + f'{b:.2f}'+'}' + fr'{c:.2f}$'


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
        return fr'$({a:.2E})\times x^4 + ({b:.2E})\times x^3 + ({c:.2E})\times x^2 + {d:.2f}\times x + {e:.2f}$'


def fit_pred(df:pd.DataFrame, fit_func:AbsFunc, centre:str='med', width:str='c68', bins:np.ndarray=np.linspace(100,8000,80)) -> pd.DataFrame:
    r'''
    Fits chosen function to centre of predictions in bins of true energy, considering width as uncertinties

    Arguments:
        df: DataFrame of predictions and targets
        fit_func: :class:`~muon_regression.plotting.fitting.AbsFunc` to fit targets to predictions
        centre: 'med' or 'mean' for central values
        width: 'c68' of 'std' for uncertainty
        bins: bin edges for true energy
    '''
    
    def _percentile(n:float) -> Callable[[np.ndarray], float]:
        def __percentile(x): return np.nanpercentile(x, n)
        __percentile.__name__ = f'{n}'
        return __percentile

    grps = df.groupby(pd.cut(df.gen_target, bins))
    agg = grps.agg(pred_med=pd.NamedAgg(column='pred',  aggfunc='median'),
                   pred_mean=pd.NamedAgg(column='pred', aggfunc='mean'),
                   pred_std=pd.NamedAgg(column='pred',  aggfunc='std'),
                   pred_p16=pd.NamedAgg(column='pred',  aggfunc=_percentile(15.865)),
                   pred_p84=pd.NamedAgg(column='pred',  aggfunc=_percentile(84.135))).reset_index()
    agg['pred_c68'] = (agg.pred_p84-agg.pred_p16)/2
    agg['gen_target'] = [e.left+((e.right-e.left)/2) for e in agg.gen_target.values]
    
    fit_func.fit(agg.gen_target, agg[f'pred_{centre}'], sigma=agg[f'pred_{width}'])
    return agg
    