import pandas as pd
import numpy as np
from typing import List, Callable

__all__ = ['get_binned_errors', 'get_res']


def get_binned_errors(df:pd.DataFrame, bins:np.ndarray=np.linspace(100,8000,80), percentiles:List[float]=[2.5,15.865,50,84.135,97.5]) -> pd.DataFrame:
    r'''
    Computed percentiles of regression errors in bins of true energy.

    Arguments:
        df: DataFrame with truen and predicted energy per muon
        bins: Array of bin edges
        percentiles: list of percentiles to compute

    Returns:
        Aggregated DataFrame with precomputed percentiles of error in bins of true energy
    '''
    
    def _percentile(n:float) -> Callable[[np.ndarray], float]:
        def __percentile(x): return np.percentile(x, n)
        __percentile.__name__ = f'{n}'
        return __percentile
    
    df['error'] = df.pred-df.gen_target
    df['sqr_error'] = df['error'].values**2
    df['rel_error'] = df.error.values/df.gen_target.values

    grps = df.groupby(pd.cut(df.gen_target, bins))
    agg = grps.agg({f:[_percentile(n) for n in percentiles]+['mean', 'std', 'var'] for f in ['error', 'rel_error', 'pred', 'sqr_error']})
    agg.columns = ['_'.join(c).strip() for c in agg.columns.values]
    agg.reset_index(inplace=True)
    agg.gen_target = [e.left+((e.right-e.left)/2) for e in agg.gen_target.values]
    return agg


def get_res(df:pd.DataFrame, bins:np.ndarray=np.linspace(100,8000,41), pred_name:str='pred_corr', extra:bool=True) -> pd.DataFrame:
    r'''
    Computes resolution in bins of true energy based on predictions which are already bias-corrected

    Arguments:
        df: DataFrame with true and corrected predicted energy per muon
        bins: Array of bin edges
        extra: whether to include all information, or just the values required to compute the resolution

    Returns:
        Aggregated DataFrame with resolution in bins of true energy
    '''
    
    def _percentile(n:float) -> Callable[[np.ndarray], float]:
        def __percentile(x): return np.nanpercentile(x, n)
        __percentile.__name__ = f'{n}'
        return __percentile
    
    grps = df.groupby(pd.cut(df.gen_target, bins))
    df['error'] = df[pred_name].values/df.gen_target.values
    agg_func = dict(corr_res_p16=pd.NamedAgg(column='error', aggfunc=_percentile(15.865)),
                    corr_res_p84=pd.NamedAgg(column='error', aggfunc=_percentile(84.135)),
                    corr_pred_p16=pd.NamedAgg(column=pred_name, aggfunc=_percentile(15.865)),
                    corr_pred_p84=pd.NamedAgg(column=pred_name, aggfunc=_percentile(84.135)),
                    corr_pred_median=pd.NamedAgg(column=pred_name, aggfunc='median'),
                    true_median=pd.NamedAgg(column='gen_target', aggfunc='median'))
    if extra:
        extra_agg_func = dict(corr_res_var=pd.NamedAgg(column='error', aggfunc='var'),
                              corr_res_std=pd.NamedAgg(column='error', aggfunc='std'),
                              corr_pred_mean=pd.NamedAgg(column=pred_name, aggfunc='mean'),
                              corr_pred_var=pd.NamedAgg(column=pred_name, aggfunc='var'),
                              corr_pred_std=pd.NamedAgg(column=pred_name, aggfunc='std'),
                              corr_pred_p16=pd.NamedAgg(column=pred_name, aggfunc=_percentile(15.865)),
                              corr_pred_p84=pd.NamedAgg(column=pred_name, aggfunc=_percentile(84.135)),
                              true_mean=pd.NamedAgg(column='gen_target', aggfunc='mean'))
        agg_func = {**agg_func, **extra_agg_func}
    
    agg = grps.agg(**agg_func).reset_index()
    agg['corr_res_c68'] = (agg.corr_res_p84-agg.corr_res_p16)/2
    agg['corr_pred_c68'] = (agg.corr_pred_p84-agg.corr_pred_p16)/2
    agg['corr_rmse_median'] = np.sqrt((agg.corr_pred_c68**2)+((agg.corr_pred_median-agg.true_median)**2))
    agg['corr_frac_rmse_median'] = agg['corr_rmse_median'].values/agg.true_median.values
    if extra:
        agg['corr_rmse_mean'] = np.sqrt((agg.corr_pred_std**2)+((agg.corr_pred_mean-agg.true_mean)**2))
        agg['corr_frac_rmse_mean'] = agg['corr_rmse_mean'].values/agg.true_mean.values
    agg['gen_target'] = [e.left+((e.right-e.left)/2) for e in agg.gen_target.values]
    return agg
    