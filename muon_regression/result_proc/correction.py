import pandas as pd
import numpy as np

__all__ = ['get_bias_correction_map', 'correct_pred']


def get_bias_correction_map(df:pd.DataFrame, nbins:int) -> pd.DataFrame:
    r'''
    Computes the bias on predictions in bins of predictions.

    Arguments:
        df: DataFrame with pred and gen_target columns
        nbins: nbins for prediceted energy

    Returns:
        DataFrame with bins of predicted energy (and centre points) and mean true energy within each bin
    '''

    grps = df.groupby(pd.cut(df.pred, np.linspace(df.pred.min(),df.pred.max(),nbins)))
    agg = grps.agg(mean_true=pd.NamedAgg(column='gen_target', aggfunc='mean')).reset_index()
    agg['pred_centre'] = [e.left+((e.right-e.left)/2) for e in agg.pred.values]
    return agg


def correct_pred(df:pd.DataFrame, bias_map:pd.DataFrame) -> None:
    r'''
    Bias-corrects predictions based on the map produced by :meth:`~muon_regression.result_proc.correction.get_bias_correction_map`.
    Adds corrected predictions as a new column ('pred_corr') in `df`.

    Arguments:
        df: DataFrame with predictions in 'pred' column
        bias_map: map produced by :meth:`~muon_regression.result_proc.correction.get_bias_correction_map`.
    '''

    df['pred_corr'] = df.pred
    for b,c,t in zip(bias_map.pred,bias_map.pred_centre,bias_map.mean_true):
        df.loc[(df.pred >= b.left) & (df.pred < b.right), 'pred_corr'] += t-c
