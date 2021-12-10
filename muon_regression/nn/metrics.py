import numpy as np
import math
from typing import Optional, Tuple, Callable, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lumin.nn.metrics.eval_metric import EvalMetric
from lumin.utils.multiprocessing import mp_run

from ..result_proc import get_res
from ..plotting.fitting import AbsFunc, fit_pred

__all__ = ['ImprovRMSE', 'MaxRMSE']
        

class ImprovRMSE(EvalMetric):
    def __init__(self, partial_fit_func:Callable[[],AbsFunc], restricted_energy:bool, k:float=2e-4, n_bins:int=20, bootstrap_n:Optional[int]=None,
                 main_metric:bool=True):
        r'''
        Computes the overall improvement in RMSE over tracker due to the calo measurement.

        Arguments:
            partial_fit_func: partial of fitting class
            restricted_energy: whether to cut and fit upto 4TeV 
            k: multiplicative coefficient for tracking resolution scaling
            n_bins: The number of bins to pass to :meth:`~muon_regression.result_proc.binning.get_res`
            bootstrap_n: if not None, will compute boostrapped resolution
            main_metric: whether to consider this metric when tracking SaveBest and EarlyStopping
        '''
        
        name = 'improv_rmse'
        if restricted_energy: name += '_4tev'
        if bootstrap_n is not None: name += '_bs'
        super().__init__(name=name, lower_metric_better=False, main_metric=main_metric)
        self.restricted_energy,self.k,self.partial_fit_func,self.bootstrap_n = restricted_energy,k,partial_fit_func,bootstrap_n
        if self.restricted_energy:
            self.fit_bins = np.linspace(100,4000,40)
            self.res_bins = np.linspace(100,4000,n_bins+1)
        else:
            self.fit_bins = np.linspace(100,8000,80)
            self.res_bins = np.linspace(100,8000,n_bins+1)
        self.tracker_res = self.k*np.array([self.res_bins[i]+(self.res_bins[i+1]-self.res_bins[i])/2 for i in range(n_bins)])
        self.fit,self.fit_std = [],[]
        self.n_fit_params = self.partial_fit_func().n_params

    def evaluate(self) -> float:
        def compute_res(df:pd.DataFrame) -> Tuple[float,List[float]]:
            fit = self.partial_fit_func()
            fit_pred(df, fit_func=fit, bins=self.fit_bins)
            df['pred_corr'] = fit.inv_func(df.pred)
            res = get_res(df, bins=self.res_bins, extra=False)
            rmse = np.sqrt(1/((1/(res.corr_frac_rmse_median**2))+(1/(self.tracker_res**2))))
            improv = (self.tracker_res-rmse).mean()
            return improv, fit.params
        
        def mp_wrapper(args, out_q):
            try:    improv,fit = compute_res(args['df'])
            except: improv,fit = np.NaN,[np.NaN for _ in range(self.n_fit_params)]  # noqa E772
            out_q.put({f"{args['name']}_improv":improv, f"{args['name']}_fit":fit})
            
        df = self.get_df()
        if self.restricted_energy: df = df[(df.gen_target >= 100) & (df.gen_target <= 4000)]
        improv = None
        try:  # Fit might fail
            if self.bootstrap_n is None:
                improv,_ = compute_res(df)
            else:
                args = []
                for i in range(self.bootstrap_n): args.append({'name': i, 'df':df.sample(frac=1, replace=True, random_state=i)})
                bs = mp_run(args, mp_wrapper)
                bs_improv = np.array([bs[k] for k in bs if 'improv' in k])
                bs_fit = np.array([bs[k] for k in bs if 'fit' in k])
                self.fit.append(np.nanmean(bs_fit,0))
                self.fit_std.append(np.nanstd(bs_fit,0))
                improv = np.nanmean(bs_improv)
                del args
        except:  # noqa E722
            pass
        finally:  # Res might be NaN
            if improv is np.NaN or improv is None: improv = 0
        return improv

    def on_train_end(self) -> None:
        if self.bootstrap_n is None: return
        self.fit,self.fit_std = np.array(self.fit),np.array(self.fit_std)
        with sns.axes_style(**self.plot_settings.style), sns.color_palette(self.plot_settings.cat_palette) as palette:
            for p in range(self.n_fit_params):
                fig = plt.figure(figsize=(self.plot_settings.w_mid, self.plot_settings.h_mid))
                plt.step(range(1,len(self.fit)+1), self.fit[:,p], where='mid', color=palette[0])
                plt.errorbar(range(1,len(self.fit)+1), self.fit[:,p], yerr=self.fit_std[:,p], color=palette[0], fmt='o')
                plt.xticks(fontsize=self.plot_settings.tk_sz, color=self.plot_settings.tk_col)
                plt.yticks(fontsize=self.plot_settings.tk_sz, color=self.plot_settings.tk_col)
                plt.xlabel("Epoch", fontsize=self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
                plt.ylabel(f"{p} Param", fontsize=self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
                plt.savefig(self.model.fit_params.cb_savepath/f'fit_{p}_history{self.plot_settings.format}', bbox_inches='tight')
                plt.close(fig)


class MaxRMSE(EvalMetric):
    def __init__(self, partial_fit_func:Callable[[],AbsFunc], restricted_energy:bool, k:float=2e-4, n_bins:int=20, bootstrap_n:Optional[int]=None,
                 main_metric:bool=True):
        r'''
        Computes maximum vale of the RMSE of the combine tracker and calo measurements.

        Arguments:
            partial_fit_func: partial of fitting class
            restricted_energy: whether to cut and fit upto 4TeV 
            k: multiplicative coefficient for tracking resolution scaling
            n_bins: The number of bins to pass to :meth:`~muon_regression.result_proc.binning.get_res`
            bootstrap_n: if not None, will compute boostrapped resolution
            main_metric: whether to consider this metric when tracking SaveBest and EarlyStopping
        '''
        
        name = 'max_rmse'
        if restricted_energy: name += '_4tev'
        if bootstrap_n is not None: name += '_bs'
        super().__init__(name=name, lower_metric_better=True, main_metric=main_metric)
        self.restricted_energy,self.k,self.partial_fit_func,self.bootstrap_n = restricted_energy,k,partial_fit_func,bootstrap_n
        if self.restricted_energy:
            self.fit_bins = np.linspace(100,4000,40)
            self.res_bins = np.linspace(100,4000,n_bins+1)
        else:
            self.fit_bins = np.linspace(100,8000,80)
            self.res_bins = np.linspace(100,8000,n_bins+1)
        self.tracker_res = self.k*np.array([self.res_bins[i]+(self.res_bins[i+1]-self.res_bins[i])/2 for i in range(n_bins)])
        self.fit,self.fit_std = [],[]
        self.n_fit_params = self.partial_fit_func().n_params

    def evaluate(self) -> float:
        def compute_res(df:pd.DataFrame) -> Tuple[float,List[float]]:
            fit = self.partial_fit_func()
            fit_pred(df, fit_func=fit, bins=self.fit_bins)
            df['pred_corr'] = fit.inv_func(df.pred)
            res = get_res(df, bins=self.res_bins, extra=False)
            rmse = np.sqrt(1/((1/(res.corr_frac_rmse_median**2))+(1/(self.tracker_res**2))))
            return rmse.max(), fit.params
        
        def mp_wrapper(args, out_q):
            try:    improv,fit = compute_res(args['df'])
            except: improv,fit = np.NaN,[np.NaN for _ in range(self.n_fit_params)]  # noqa E772
            out_q.put({f"{args['name']}_improv":improv, f"{args['name']}_fit":fit})
            
        df = self.get_df()
        if self.restricted_energy: df = df[(df.gen_target >= 100) & (df.gen_target <= 4000)]
        improv = None
        try:  # Fit might fail
            if self.bootstrap_n is None:
                improv,_ = compute_res(df)
            else:
                args = []
                for i in range(self.bootstrap_n): args.append({'name': i, 'df':df.sample(frac=1, replace=True, random_state=i)})
                bs = mp_run(args, mp_wrapper)
                bs_improv = np.array([bs[k] for k in bs if 'improv' in k])
                bs_fit = np.array([bs[k] for k in bs if 'fit' in k])
                self.fit.append(np.nanmean(bs_fit,0))
                self.fit_std.append(np.nanstd(bs_fit,0))
                improv = np.nanmean(bs_improv)
                del args
        except:  # noqa E722
            pass
        finally:  # Res might be NaN
            if improv is np.NaN or improv is None: improv = math.inf
        return improv

    def on_train_end(self) -> None:
        if self.bootstrap_n is None: return
        self.fit,self.fit_std = np.array(self.fit),np.array(self.fit_std)
        with sns.axes_style(**self.plot_settings.style), sns.color_palette(self.plot_settings.cat_palette) as palette:
            for p in range(self.n_fit_params):
                fig = plt.figure(figsize=(self.plot_settings.w_mid, self.plot_settings.h_mid))
                plt.step(range(1,len(self.fit)+1), self.fit[:,p], where='mid', color=palette[0])
                plt.errorbar(range(1,len(self.fit)+1), self.fit[:,p], yerr=self.fit_std[:,p], color=palette[0], fmt='o')
                plt.xticks(fontsize=self.plot_settings.tk_sz, color=self.plot_settings.tk_col)
                plt.yticks(fontsize=self.plot_settings.tk_sz, color=self.plot_settings.tk_col)
                plt.xlabel("Epoch", fontsize=self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
                plt.ylabel(f"{p} Param", fontsize=self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
                plt.savefig(self.model.fit_params.cb_savepath/f'fit_{p}_history{self.plot_settings.format}', bbox_inches='tight')
                plt.close(fig)
