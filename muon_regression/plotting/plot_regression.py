import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from functools import partial
from typing import Optional, Dict, Union, Tuple
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D

from lumin.plotting.plot_settings import PlotSettings

from .fitting import AbsFunc, fit_pred
from ..experiment.exp_journal import ExpJournal
from ..result_proc.binning import get_res, get_binned_errors

__all__ = ['load_df', 'plot_reg', 'plot_res_reg', 'plot_res_trueE', 'plot_loss', 'plot_sqrt_var', 'plot_binned_error', 'plot_binned_errors',
           'plot_binned_error_stds', 'plot_binned_corr_res', 'plot_true_v_pred', 'plot_pred_v_true', 'plot_confidence', 'plot_slices', 'plot_combined_res',
           'plot_binned_rmse', 'produce_plots', 'plot_combined_rmse', 'plot_test_pred_v_true', 'plot_frac_rmse']


def load_df(path):
    
    r'''
    Load dataframe with results

    Arguments:
        path: filepath to csv file
    Returns:
        pandas frame with results
    '''
    return pd.read_csv(path)


def plot_reg(df, save_path=''):
    
    r'''
    Plot regression of the predicted energies

    Arguments:
        df: input pandas frame
        save_path: output path for the .pdf file
    '''
    
    h = sns.jointplot(x="gen_target", y="pred", data=df, kind='scatter', 
                      s=2, alpha=0.05, marginal_kws={"kde":False})
    h.set_axis_labels('True Energy [GeV]', 'Predicted Energy [GeV]', fontsize=16)
    # plt.show()
    plt.savefig(save_path+'regression.pdf', bbox_inches='tight')
    

def plot_res_reg(df, save_path=''):
    
    r'''
    Plot residuals of the predicted energies to the regression fit

    Arguments:
        df: input pandas frame
        save_path: output path for the .pdf file
    '''
    
    h = sns.jointplot(x="gen_target", y="pred", data=df, kind='resid', scatter_kws={"s": 2, 'alpha':0.05}, marginal_kws={"kde":False})
    h.set_axis_labels('True Energy [GeV]', 'Residuals [GeV]', fontsize=16)
    # plt.show()
    plt.savefig(save_path+'residuals_regression.pdf', bbox_inches='tight')
    
    
def plot_res_trueE(df, save_path=''):

    r'''
    Plot residuals of the predicted energies to the true energies

    Arguments:
        df: input pandas frame
        save_path: output path for the .pdf file
    '''
    
    df['res'] = df['gen_target'] - df['pred']
    h = sns.jointplot(x="gen_target", y="res", data=df, s=2, alpha=0.05)
    h.set_axis_labels('True Energy [GeV]', 'Residuals [GeV]', fontsize=16)
    plt.figtext(0.2,0.75, "residual mean: " + str(round(df.res.mean(),2)))
    plt.figtext(0.2,0.72, "residual std: " + str(round(df.res.std(),2)))
    # plt.show()
    plt.savefig(save_path+'residuals_trueE.pdf', bbox_inches='tight')

    
def plot_loss(df, loss_name, save_path=''):
    
    r'''
    Plot loss as a function of the true energy

    Arguments:
        df: input pandas frame
        loss_name: string name of loss column
        save_path: string output path for the .pdf file
    '''
    
    h = sns.jointplot(x="gen_target", y=loss_name, data=df, s=2, scatter_kws={'alpha':0.05})
    h.set_axis_labels('True Energy [GeV]', r'$\Delta E^2 / E$', fontsize=16)
    # plt.show()
    plt.savefig(save_path+loss_name + '_loss.pdf', bbox_inches='tight')
    
    
def get_sqrt_var(df, bin_edges):
    
    r'''
    Calculate sqrt(variance ( (pred-truth)/truth) ) ) in a give bin

    Arguments:
        df: input pandas frame
        bin_edges: tuple with bin edges
    Returns
        square root of variance
    '''    
    
    e_bin = df[(df['gen_target'] > bin_edges[0]) & (df['gen_target'] < bin_edges[1])]
    var = np.var((e_bin['pred'] - e_bin['gen_target']) / e_bin['gen_target'])
    return np.sqrt(var)
    

def plot_sqrt_var(df, n_bins=10, save_path=''):
    
    r'''
    Plot sqrt(variance ( (pred-truth)/truth) ) )

    Arguments:
        df: input pandas frame
        n_bins: int number of bins
        save_path: string output path for the .pdf file
    '''
    
    bin_edges = np.histogram_bin_edges(df['gen_target'], bins=10)
    center = (bin_edges[:-1] + bin_edges[1:]) / 2
    bins = list(zip(bin_edges[:-1], bin_edges[1:]))
    sqrt_var = [get_sqrt_var(df, b) for b in bins]
    
    plt.plot(center, sqrt_var)
    plt.ylabel(r'$\sqrt{ Var( (E_{pred} - E_{true}) / E_{true} ) }$')
    plt.xlabel('True Energy [GeV]')
    plt.savefig(save_path+'sqrt_var.pdf', bbox_inches='tight')
    

def plot_binned_error(df:pd.DataFrame, agg_df:pd.DataFrame, relative:bool, savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
    r'''
    Plots the (relative) error on predicted energy of muons as in bins of the true energy, as well as the specified percentiles.

    Arguments:
        df: DataFrame with truen and predicted energy per muon
        agg_df: Aggregated DataFrame with precomputed percentiles of error in bins of true energy
        relative: Whether to plot relative errors
        savename: Optional name of file to which to save the plot of feature importances
        settings: `PlotSettings` class to control figure appearance
    '''    

    with sns.axes_style(**settings.style), sns.color_palette(settings.cat_palette) as palette:
        plt.figure(figsize=(settings.w_mid, settings.h_mid))
        feat = 'rel_error' if relative else 'error'
        plt.scatter(df.gen_target, df[feat], color=palette[0], marker='o', alpha=0.05, label=None)
        plot = partial(plt.plot, agg_df.gen_target, color=palette[1])
        plot(agg_df[f'{feat}_2.5'], linestyle=':',  label='±2 sigma')
        plot(agg_df[f'{feat}_15.865'], linestyle='--', label='±1 sigma')
        plot(agg_df[f'{feat}_50'],                 label='Median')
        plot(agg_df[f'{feat}_84.135'], linestyle='--', label='')
        plot(agg_df[f'{feat}_97.5'], linestyle=':',  label='')
        
        plt.ylim((df[feat].quantile(.005), df[feat].quantile(.996)))
        plt.xlabel(r'$E_{\mathrm{True}}\ [GeV]$', fontsize=settings.lbl_sz, color=settings.lbl_col)
        ylbl = r'$\frac{E_{\mathrm{Pred.}}-E_{\mathrm{True}}}{E_{\mathrm{True}}}$' if relative else r'$E_{\mathrm{Pred.}}-E_{\mathrm{True}}\ [GeV]$'
        plt.ylabel(ylbl, fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.legend(loc=settings.leg_loc, fontsize=settings.leg_sz)
        plt.title(settings.title, fontsize=settings.title_sz, color=settings.title_col, loc=settings.title_loc)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}', bbox_inches='tight')
        plt.show()


def plot_binned_errors(agg_dfs:Dict[int,pd.DataFrame], relative:bool, savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
    r'''
    Plots the (relative) mean error on predicted energy of muons as in bins of the true energy for multiple runs.

    Arguments:
        agg_dfs: Dictionary of aggregated DataFrame with precomputed percentiles of error in bins of true energy
        relative: Whether to plot relative errors
        savename: Optional name of file to which to save the plot of feature importances
        settings: `PlotSettings` class to control figure appearance
    '''    

    with sns.axes_style(**settings.style), sns.color_palette(settings.cat_palette):
        plt.figure(figsize=(settings.w_mid, settings.h_mid))
        feat = 'rel_error' if relative else 'error'
        for i in agg_dfs:
            plt.plot(agg_dfs[i].gen_target, np.abs(agg_dfs[i][f'{feat}_mean']), label=f'{i}')
        
        plt.xlabel(r'$E_{\mathrm{True}}\ [GeV]$', fontsize=settings.lbl_sz, color=settings.lbl_col)
        ylbl = r'$\left|\frac{E_{\mathrm{True}}-E_{\mathrm{Pred.}}}{E_{\mathrm{True}}}\right|$' if relative else r'$\left|E_{\mathrm{True}}-E_{\mathrm{Pred.}}\right|\ [GeV]$'
        plt.ylabel(ylbl, fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.legend(loc=settings.leg_loc, fontsize=settings.leg_sz)
        plt.title(settings.title, fontsize=settings.title_sz, color=settings.title_col, loc=settings.title_loc)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}', bbox_inches='tight')
        plt.show()
    

def plot_binned_error_stds(agg_dfs:Dict[int,pd.DataFrame], savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
    r'''
    Plots the standard deviation of relative errors for multiple runs

    Arguments:
        agg_dfs: Dictionary of aggregated DataFrame with precomputed percentiles of error in bins of true energy
        relative: Whether to plot relative errors
        savename: Optional name of file to which to save the plot of feature importances
        settings: `PlotSettings` class to control figure appearance
    '''    

    with sns.axes_style(**settings.style), sns.color_palette(settings.cat_palette):
        plt.figure(figsize=(settings.w_mid, settings.h_mid))
        for i in agg_dfs:
            plt.plot(agg_dfs[i].gen_target, agg_dfs[i]['rel_error_std'], label=f'{i}')
        
        plt.xlabel(r'$E_{\mathrm{True}}\ [GeV]$', fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.ylabel(r'$\sigma\left[\frac{E_{\mathrm{True}}-E_{\mathrm{Pred.}}}{E_{\mathrm{True}}}\right]$', fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.legend(loc=settings.leg_loc, fontsize=settings.leg_sz)
        plt.title(settings.title, fontsize=settings.title_sz, color=settings.title_col, loc=settings.title_loc)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}', bbox_inches='tight')
        plt.show()


def plot_binned_corr_res(agg_dfs:Union[pd.DataFrame,Dict[int,pd.DataFrame]], width:str='c68',
                         savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> Tuple[float,float]:
    r'''
    Plots the standard deviation or variance of corrected relative errors for multiple runs

    Arguments:
        agg_dfs: Dictionary of aggregated DataFrame with precomputed percentiles of error in bins of true energy
        width: Width to plot, std, var, c68
        savename: Optional name of file to which to save the plot of feature importances
        settings: `PlotSettings` class to control figure appearance

    Returns:
        Resolution at 1 TeV and 3 TeV
    ''' 

    if not isinstance(agg_dfs, dict): agg_dfs = {0:agg_dfs}
    with sns.axes_style(**settings.style), sns.color_palette(settings.cat_palette):
        plt.figure(figsize=(settings.w_mid, settings.h_mid))
        for i in agg_dfs:
            rs = []
            for e in [1000,3000]:
                try: rs.append(interp1d(agg_dfs[i].gen_target, agg_dfs[i][f'corr_res_{width}'])(e))
                except ValueError: rs.append(np.NaN)
            plt.plot(agg_dfs[i].gen_target, agg_dfs[i][f'corr_res_{width}'], label=f'ID{i}: {rs[0]:.2f}@1TeV {rs[1]:.2f}@3TeV')
        
        plt.xlabel(r'$E_{\mathrm{True}}\ [GeV]$', fontsize=settings.lbl_sz, color=settings.lbl_col)
        if width == 'var':   sigma = r'$\sigma^2'
        elif width == 'std': sigma = r'$\sigma'
        elif width == 'c68': sigma = r'Central $68\%'
        plt.ylabel(sigma+r'\left[\frac{E_{P,\mathrm{corr.}}}{E_T}\right]$', fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.legend(loc=settings.leg_loc, fontsize=settings.leg_sz)
        plt.title(settings.title, fontsize=settings.title_sz, color=settings.title_col, loc=settings.title_loc)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}', bbox_inches='tight')
        plt.show()
    return rs


def plot_true_v_pred(df:pd.DataFrame, corr_pred:bool, fit_func:Optional[AbsFunc]=None,
                     savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
    r'''
    Plots the distribution of (bias-corrected) predicted energy  as a function of true energy

    Arguments:
        df: DataFrame with true and predicted energy per muon
        corr_pred: Whether to plot the corrected predictions
        savename: Optional name of file to which to save the plot of feature importances
        settings: `PlotSettings` class to control figure appearance
    ''' 

    with sns.axes_style(**settings.style), sns.color_palette(settings.cat_palette):
        plt.figure(figsize=(settings.w_mid, settings.h_mid))        
        sns.regplot('pred_corr' if corr_pred else 'pred', 'gen_target',  df, fit_reg=False, scatter_kws={'alpha':0.05})
        if fit_func is not None:
            x = np.linspace(0,df.gen_target.max()+1000,100)
            y = fit_func.inv_func(x)
            plt.plot(x,y, label='Inverse of fit')
        xlbl = r'$E_{\mathrm{Pred}}\ [GeV]$'
        if corr_pred: xlbl = 'Corrected ' + xlbl
        plt.xlabel(xlbl, fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.ylabel(r'$E_{\mathrm{True}}\ [GeV]$', fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        if fit_func is not None:
            plt.ylim((0,df.gen_target.max()+1000))
            plt.legend(fontsize=settings.leg_sz)
        plt.title(settings.title, fontsize=settings.title_sz, color=settings.title_col, loc=settings.title_loc)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}', bbox_inches='tight')
        plt.show()


def plot_pred_v_true(df:pd.DataFrame, agg_df:pd.DataFrame,
                     fit_func:Optional[AbsFunc]=None, savename:Optional[str]=None,
                     settings:PlotSettings=PlotSettings()) -> None:
    r'''
    Plots the distribution of (bias-corrected) predicted energy  as a function of true energy

    Arguments:
        df: DataFrame with true and predicted energy per muon
        agg_df: Aggregated DataFrame with precomputed percentiles of error in bins of true energy
        fit: Power-law fit parameters
        savename: Optional name of file to which to save the plot of feature importances
        settings: `PlotSettings` class to control figure appearance
    '''

    with sns.axes_style(**settings.style), sns.color_palette(settings.cat_palette):
        plt.figure(figsize=(settings.w_mid, settings.h_mid))        
        sns.regplot('gen_target', 'pred_corr' if fit_func is None else 'pred', df, fit_reg=False, scatter_kws={'alpha':0.05})
        x_max = df.gen_target.max()
        if fit_func is not None:
            x = np.linspace(100,x_max,200)
            y = fit_func.func(x)
            plt.plot(x,y, label=fit_func.string())
            plt.ylim((100,x_max+1000))
        else:
            plt.plot([100,x_max],[100,x_max], linestyle='--', color='black')

        ylbl = r'$E_{\mathrm{Pred}}\ [GeV]$'
        if fit_func is None: ylbl = 'Corrected ' + ylbl
        plt.ylabel(ylbl, fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.xlabel(r'$E_{\mathrm{True}}\ [GeV]$', fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        if fit_func is not None: plt.legend(fontsize=settings.leg_sz)
        plt.title(settings.title, fontsize=settings.title_sz, color=settings.title_col, loc=settings.title_loc)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}', bbox_inches='tight')
        plt.show()

        
def plot_test_pred_v_true(df:pd.DataFrame, agg_df:pd.DataFrame, showfliers:bool=False, savename:Optional[str]=None,
                          settings:PlotSettings=PlotSettings()) -> None:
    r'''
    Plots the distribution of bias-corrected predicted energy as a function of true energy for test data

    Arguments:
        df: DataFrame with true and predicted energy per muon
        agg_df: Aggregated DataFrame with precomputed percentiles of error in bins of true energy
        showfliers: Whether to plot outlier datapoints
        savename: Optional name of file to which to save the plot of feature importances
        settings: `PlotSettings` class to control figure appearance
    '''

    with sns.axes_style(**settings.style), sns.color_palette(settings.cat_palette):
        plt.figure(figsize=(settings.w_mid, settings.h_mid))        
        
        plt.plot([0,df.gen_target.nunique()-1],[0,df.gen_target.max()], linestyle='--', color='black')
        sns.boxenplot(x='gen_target', y='pred_corr', data=df, color='orange', showfliers=showfliers)

        ylbl = r'Corrected $E_{\mathrm{Pred}}\ [GeV]$'
        plt.ylabel(ylbl, fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.xlabel(r'$E_{\mathrm{True}}\ [GeV]$', fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.title(settings.title, fontsize=settings.title_sz, color=settings.title_col, loc=settings.title_loc)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}', bbox_inches='tight')
        plt.show()

        
def plot_confidence(agg_df:pd.DataFrame, savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
    r'''
    Plots the standard deviation or variance of corrected predictions for multiple runs

    Arguments:
        agg_df: Dictionary of aggregated DataFrame with precomputed percentiles of error in bins of true energy
        savename: Optional name of file to which to save the plot of feature importances
        settings: `PlotSettings` class to control figure appearance
    ''' 
    
    with sns.axes_style(**settings.style), sns.color_palette(settings.cat_palette) as palette:
        plt.figure(figsize=(settings.w_mid, settings.h_mid))
        plt.plot([100,8000],[100,8000], linestyle='--', color='black')
        plt.fill_between(agg_df.gen_target, agg_df.corr_pred_p84, agg_df.corr_pred_p16, color=palette[0], alpha=0.5)
        us,ms,ds = [],[],[]
        for e in [1000,3000]:
            try:
                us.append(interp1d(agg_df.gen_target, agg_df.corr_pred_p84)(e))
                ms.append(interp1d(agg_df.gen_target, agg_df.corr_pred_median)(e))
                ds.append(interp1d(agg_df.gen_target, agg_df.corr_pred_p16)(e))
            except ValueError:
                us.append(np.NaN)
                ms.append(np.NaN)
                ds.append(np.NaN)
        plt.plot(agg_df.gen_target, agg_df.corr_pred_median, color=palette[0],
                 label=f'+{us[0]-ms[0]:.0f}/-{ms[0]-ds[0]:.0f}@1TeV +{us[1]-ms[1]:.0f}/-{ms[1]-ds[1]:.0f}@3TeV')

        plt.xlabel(r'$E_{\mathrm{True}}\ [GeV]$',  fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.ylabel(r'Corrected $E_{\mathrm{Pred.}}\ [GeV]$', fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.legend(loc=settings.leg_loc, fontsize=settings.leg_sz)
        plt.title(settings.title, fontsize=settings.title_sz, color=settings.title_col, loc=settings.title_loc)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}', bbox_inches='tight')
        plt.show()


def plot_slices(df:pd.DataFrame, corr_pred:bool, n_bins:int=80, savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
    r'''
    Plots the distribution of (bias-corrected) predicted energy  as a function of true energy

    Arguments:
        df: DataFrame with true and predicted energy per muon
        corr_pred: Whether to use the corrected predictions
        savename: Optional name of file to which to save the plot of feature importances
        settings: `PlotSettings` class to control figure appearance
    '''
    
    df['pred/true'] = df['pred_corr' if corr_pred else 'pred']/df.gen_target
    bins = np.linspace(100, 8000 if df.gen_target.max() > 4000 else 4000, n_bins+1)

    with sns.axes_style(**settings.style), sns.color_palette(settings.cat_palette):
        nf = 10*n_bins//10
        fig,axs = plt.subplots(n_bins//10, 10, figsize=(2*settings.w_mid, 2*settings.h_mid))
        for i,ax in enumerate(axs.flatten()):
            if i == n_bins-1: break
            x = df.loc[(df['gen_target'] >= bins[i]) & (df['gen_target'] < bins[i+1]), 'pred/true']
            sns.kdeplot(x, label=f'{bins[i]:.0f}-{bins[i+1]:.0f}', ax=ax)
            ax.xaxis.set_ticks(np.arange(0,3,0.5))
            ax.tick_params(axis='x', rotation=45)
            ax.set_xlim((0,3))
            if i == 7*nf//8: ax.set_xlabel('Corrected pred/true' if corr_pred else 'pred/true', fontsize=settings.lbl_sz, color=settings.lbl_col)
        if savename is not None: fig.savefig(settings.savepath/f'{savename}{settings.format}', bbox_inches='tight')
        plt.show()


def plot_test_slices(df:pd.DataFrame, corr_pred:bool, bins:np.ndarray, savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
    r'''
    Plots the distribution of (bias-corrected) predicted energy  as a function of true energy

    Arguments:
        df: DataFrame with true and predicted energy per muon
        corr_pred: Whether to use the corrected predictions
        savename: Optional name of file to which to save the plot of feature importances
        settings: `PlotSettings` class to control figure appearance
    '''
    df['pred/true'] = df['pred_corr' if corr_pred else 'pred']/df.gen_target
    n_bins = len(bins)-1
    with sns.axes_style(**settings.style), sns.color_palette(settings.cat_palette):
        fig,axs = plt.subplots(1, n_bins, figsize=(2*settings.w_mid, settings.h_mid), sharey=True, sharex=True)
        for i,ax in enumerate(axs.flatten()):
            dashed_line = Line2D([1.0, 1.0], [0.0, 2.0], linestyle='--', linewidth=1, color=[0,0,0], zorder=1, transform=ax.transData)
            ax.lines.append(dashed_line)
            x = df.loc[(df['gen_target'] >= bins[i]) & (df['gen_target'] < bins[i+1]), 'pred/true']
            sns.kdeplot(x, ax=ax, label='')
            ax.xaxis.set_ticks(np.arange(0,3,0.5), )
            ax.tick_params(axis='x', rotation=45)
            ax.set_xlim((0,3))
            ax.set_xlabel(f'{((bins[i+1]-bins[i])/2)+bins[i]:.0f} GeV', fontsize=settings.lbl_sz, color=settings.lbl_col)
            ax.tick_params(axis='x', labelsize=settings.tk_sz, labelcolor=settings.tk_col)
            ax.tick_params(axis='y', labelsize=settings.tk_sz, labelcolor=settings.tk_col)
            if i == 0: ax.set_ylabel('Density', fontsize=settings.lbl_sz, color=settings.lbl_col)
            if i == 4: ax.set_xlabel(f'{((bins[i+1]-bins[i])/2)+bins[i]:.0f} GeV\n' + 
                                     r'Corrected $E_{\mathrm{Pred.}}/E_{\mathrm{True}}$' if corr_pred
                                     else r'$E_{\mathrm{Pred.}}/E_{\mathrm{True}}$', fontsize=settings.lbl_sz,
                                     color=settings.lbl_col)
        if savename is not None: fig.savefig(settings.savepath/f'{savename}{settings.format}', bbox_inches='tight')
        plt.show()
        

def plot_combined_res(agg_df:pd.DataFrame, k:float=2e-4, savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> Tuple[float,float]:
    r'''
    Plots the standard deviation or variance of corrected relative errors for multiple runs

    Arguments:
        agg_df: Aaggregated DataFrame with precomputed percentiles of error in bins of true energy
        k: scaling coefficient for the tracker resolution
        savename: Optional name of file to which to save the plot of feature importances
        settings: `PlotSettings` class to control figure appearance

    Returns:
        Maximum combined resolution
    ''' 
    
    agg_df['tracker_res'] = (agg_df.gen_target)*k
    agg_df['combined_res'] = np.sqrt(1/((1/(agg_df.corr_res_c68**2))+(1/(agg_df.tracker_res**2))))
    
    max_res = agg_df['combined_res'].max()
    
    with sns.axes_style(**settings.style), sns.color_palette(settings.cat_palette):
        plt.figure(figsize=(settings.w_mid, settings.h_mid))
        plt.plot(agg_df.gen_target, 100*agg_df['corr_res_c68'], label='Calorimeter')
        plt.plot(agg_df.gen_target, 100*agg_df['tracker_res'], label='Tracker')
        plt.plot(agg_df.gen_target, 100*agg_df['combined_res'], label=f'Combined resolution, max={max_res*100:.1f}%')
        
        plt.xlabel(r'$E_{\mathrm{True}}\ [GeV]$', fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.ylabel('Resolution [%]', fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.legend(loc=settings.leg_loc, fontsize=settings.leg_sz)
        plt.title(settings.title, fontsize=settings.title_sz, color=settings.title_col, loc=settings.title_loc)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}', bbox_inches='tight')
        plt.show()
    return max_res


def plot_combined_rmse(agg_df:pd.DataFrame, k:float=2e-4, savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> Tuple[float,float]:
    r'''
    Arguments:
        agg_df: Aaggregated DataFrame with precomputed percentiles of error in bins of true energy
        k: scaling coefficient for the tracker resolution
        savename: Optional name of file to which to save the plot of feature importances
        settings: `PlotSettings` class to control figure appearance

    Returns:
        Maximum combined resolution
    ''' 
    
    agg_df['tracker_rmse'] = (agg_df.gen_target)*k
    agg_df['combined_rmse'] = np.sqrt(1/((1/(agg_df.corr_frac_rmse_median**2))+(1/(agg_df.tracker_rmse**2))))
    
    max_res = agg_df['combined_rmse'].max()
    improv = (agg_df['tracker_rmse']-agg_df['combined_rmse']).mean()
    
    with sns.axes_style(**settings.style), sns.color_palette(settings.cat_palette):
        plt.figure(figsize=(settings.w_mid, settings.h_mid))
        plt.plot(agg_df.gen_target, 100*agg_df['corr_frac_rmse_median'], label='Calorimeter')
        plt.plot(agg_df.gen_target, 100*agg_df['tracker_rmse'], label='Tracker')
        plt.plot(agg_df.gen_target, 100*agg_df['combined_rmse'], label=f'Combined frac. RMSE,\nmax value={max_res*100:.1f}%,\nmean improvement={improv*100:.1f}%')
        
        plt.xlabel(r'$E_{\mathrm{True}}\ [GeV]$', fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.ylabel('Percentage RMSE [%]', fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.legend(loc=settings.leg_loc, fontsize=settings.leg_sz)
        plt.title(settings.title, fontsize=settings.title_sz, color=settings.title_col, loc=settings.title_loc)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}', bbox_inches='tight')
        plt.show()
    return max_res,improv


def plot_binned_rmse(agg_dfs:Union[pd.DataFrame,Dict[int,pd.DataFrame]], median:bool=True, savename:Optional[str]=None,
                     settings:PlotSettings=PlotSettings()) -> None:
    r'''
    Plots the standard deviation or variance of corrected relative errors for multiple runs

    Arguments:
        agg_dfs: Dictionary of aggregated DataFrame with precomputed percentiles of error in bins of true energy
        median: Whether to use RMSE computed with c68 and median, or std and mean
        savename: Optional name of file to which to save the plot of feature importances
        settings: `PlotSettings` class to control figure appearance
    ''' 

    if not isinstance(agg_dfs, dict): agg_dfs = {0:agg_dfs}
    with sns.axes_style(**settings.style), sns.color_palette(settings.cat_palette):
        plt.figure(figsize=(settings.w_mid, settings.h_mid))
        for i in agg_dfs: plt.plot(agg_dfs[i].gen_target, agg_dfs[i][f'corr_rmse_{"median" if median else "mean"}'], label=f'{i}')
        plt.xlabel(r'$E_{\mathrm{True}}\ [GeV]$', fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.ylabel(r'RMSE: $\sqrt{\mathrm{pred.\ width}^2+\mathrm{bias}^2}$', fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        if len(agg_dfs) > 1: plt.legend(loc=settings.leg_loc, fontsize=settings.leg_sz)
        plt.title(settings.title, fontsize=settings.title_sz, color=settings.title_col, loc=settings.title_loc)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}', bbox_inches='tight')
        plt.show()


def plot_frac_rmse(agg_dfs:Union[pd.DataFrame,Dict[int,pd.DataFrame]], savename:Optional[str]=None, settings:PlotSettings=PlotSettings()) -> None:
    r'''
    Plots the standard deviation or variance of corrected relative errors for multiple runs

    Arguments:
        agg_dfs: Dictionary of aggregated DataFrame with precomputed percentiles of error in bins of true energy
        savename: Optional name of file to which to save the plot of feature importances
        settings: `PlotSettings` class to control figure appearance
    ''' 

    if not isinstance(agg_dfs, dict): agg_dfs = {0:agg_dfs}
    with sns.axes_style(**settings.style), sns.color_palette(settings.cat_palette):
        plt.figure(figsize=(settings.w_mid, settings.h_mid))
        for i in agg_dfs: plt.plot(agg_dfs[i].gen_target, 100*agg_dfs[i]['corr_frac_rmse_median'], label=f'{i}')
        plt.xlabel(r'$E_{\mathrm{True}}\ [GeV]$', fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.ylabel('Percentage RMSE [%]', fontsize=settings.lbl_sz, color=settings.lbl_col)
        plt.xticks(fontsize=settings.tk_sz, color=settings.tk_col)
        plt.yticks(fontsize=settings.tk_sz, color=settings.tk_col)
        if len(agg_dfs) > 1: plt.legend(loc=settings.leg_loc, fontsize=settings.leg_sz)
        plt.title(settings.title, fontsize=settings.title_sz, color=settings.title_col, loc=settings.title_loc)
        if savename is not None: plt.savefig(settings.savepath/f'{savename}{settings.format}', bbox_inches='tight')
        plt.show()
        

def produce_plots(df:pd.DataFrame, jrn:ExpJournal, settings:PlotSettings, fit_func:AbsFunc, restrict_range:bool) -> pd.DataFrame:
    cache_path = settings.savepath
    if restrict_range:
        df = df[df.gen_target <= 4000]
        agg_bins = np.linspace(100,4000,40)
        res_bins = np.linspace(100,4000,21)
        settings.savepath /= 'restricted_range'
        result_prfx = 'restricted_range_'
    else:
        agg_bins = np.linspace(100,8000,80)
        res_bins = np.linspace(100,8000,41)
        settings.savepath /= 'full_range'
        result_prfx = 'full_range_'
    settings.savepath.mkdir(exist_ok=True)
    savepath = str(settings.savepath) + '/'

    try:
        plot_binned_error(df, get_binned_errors(df), relative=True, savename='quantiles', settings=settings)

        # Plot regression
        plot_reg(df, save_path=savepath)
        
        # Plot residuals
        plot_res_reg(df, save_path=savepath)
        plot_res_trueE(df, save_path=savepath)

        # Fitted bias correction
        centre = 'mean'
        agg = fit_pred(df, centre=centre, fit_func=fit_func, bins=agg_bins)
        plot_pred_v_true(df, agg_df=agg, fit_func=fit_func, savename='fitted_pred', settings=settings)
        plot_true_v_pred(df, corr_pred=False, fit_func=fit_func, savename='fit_extrap', settings=settings)
        df['pred_corr'] = fit_func.inv_func(df.pred)
        res = get_res(df, bins=res_bins)
        plot_pred_v_true(df, agg_df=res, savename='corrected_prediction', settings=settings)
        resolutions = plot_binned_corr_res(res, savename='resolution_c68', settings=settings)
        jrn[f'{result_prfx}res_1Tev'] = np.nan_to_num(resolutions[0])
        jrn[f'{result_prfx}res_3Tev'] = np.nan_to_num(resolutions[1])
        plot_confidence(res, savename='confidence_c68', settings=settings)
        jrn[f'{result_prfx}max_combined_res'] = plot_combined_res(res, savename='combined_resolution', settings=settings)
        jrn[f'{result_prfx}max_combined_rmse'],jrn[f'{result_prfx}combined_rmse_improv'] = plot_combined_rmse(res, savename='combined_rmse', settings=settings)
        plot_binned_rmse(res, savename='mse', settings=settings)
        res.to_csv(f'{savepath}res.csv')
        plot_slices(df=df, corr_pred=False, n_bins=41, savename='slices', settings=settings)
        plot_slices(df=df, corr_pred=True, n_bins=41, savename='corr_slices', settings=settings)
    finally:
        settings.savepath = cache_path
    return res

