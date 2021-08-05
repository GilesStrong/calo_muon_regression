from typing import Tuple
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
import numpy as np
import math

from lumin.nn.callbacks.callback import Callback
from lumin.utils.misc import ForwardHook, BackwardHook

from torch import Tensor
import torch.nn as nn
import torch

__all__ = ['MuonConvTelemetryBodyBackwards', 'MuonConvTelemetryBodyForwards', 'MuonConvTelemetryHeadBackwards', 'MuonConvTelemetryHeadForwards']


class AbsMuonConvTelemetry(Callback):
    '''Draws from from https://github.com/fastai/course-v3/blob/master/nbs/dl2/06_cuda_cnn_hooks_init.ipynb under apache2'''
    
    def __init__(self, hist_range: Tuple[float,float]):
        super().__init__()
        self.hist_range = hist_range
        
    def plot(self) -> None:
        def _plot_stats(n):
            with sns.axes_style(**self.plot_settings.style), sns.color_palette(self.plot_settings.cat_palette):
                fig,axs = plt.subplots(1,2,figsize=(2*self.plot_settings.w_small,self.plot_settings.h_small))
                for l in self.stats:
                    ms,ss = self.stats[l][0],self.stats[l][1]
                    if n > 0: ms,ss = ms[:n],ss[:n]
                    axs[0].plot(ms, label=l)
                    axs[1].plot(ss, label=l)
                plt.legend(loc=self.plot_settings.leg_loc, fontsize=self.plot_settings.leg_sz/2)
                for i,ax in enumerate(axs):
                    ax.set_xlabel("Iteration", fontsize=self.plot_settings.lbl_sz/2, color=self.plot_settings.lbl_col)
                    ax.set_ylabel('Mean' if i == 0 else 'Stdev', fontsize=self.plot_settings.lbl_sz/2, color=self.plot_settings.lbl_col)
                    ax.tick_params(axis='x', labelsize=self.plot_settings.tk_sz/2, labelcolor=self.plot_settings.tk_col)
                    ax.tick_params(axis='y', labelsize=self.plot_settings.tk_sz/2, labelcolor=self.plot_settings.tk_col)
                plt.show()

        def _plot_hists():
            with sns.axes_style(**self.plot_settings.style):
                fig,axs = plt.subplots(len(self.stats),1, figsize=(self.plot_settings.w_mid,len(self.stats)*self.plot_settings.h_small))
                for l,ax in zip(self.stats,axs.flatten()):
                    plot_ = sns.heatmap(torch.stack(self.stats[l][2]).t().float().log1p(), xticklabels=range(len(self.stats[l][2])),
                                        yticklabels=np.linspace(*self.hist_range,41), ax=ax, cmap=self.plot_settings.seq_palette, cbar=False)
                    for ind, label in enumerate(plot_.get_yticklabels()):
                        if ind % 10 == 0: label.set_visible(True)
                        else:             label.set_visible(False)
                    for ind, label in enumerate(plot_.get_xticklabels()):
                        if ind % 10 == 0: label.set_visible(True)
                        else:             label.set_visible(False)
                    ax.set_xlabel("Iteration", fontsize=self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
                    ax.set_ylabel('Output density', fontsize=self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
                    ax.tick_params(axis='x', labelsize=self.plot_settings.tk_sz, labelcolor=self.plot_settings.tk_col)
                    ax.tick_params(axis='y', labelsize=self.plot_settings.tk_sz, labelcolor=self.plot_settings.tk_col)
                    ax.set_title(l, fontsize=self.plot_settings.title_sz, color=self.plot_settings.title_col)
                plt.tight_layout()
                plt.show()
                
        def _plot_fracs():
            with sns.axes_style(**self.plot_settings.style), sns.color_palette(self.plot_settings.cat_palette):
                plt.figure(figsize=(self.plot_settings.w_mid, self.plot_settings.h_mid))
                for l in self.stats:
                    h = torch.stack(self.stats[l][2]).t().float()
                    f = h[:math.ceil((40*(1-self.hist_range[0]))/(self.hist_range[1]-self.hist_range[0]))].sum(0)/h.sum(0)
                    plt.plot(f, label=l)
                    plt.xlabel("Iteration", fontsize=self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
                    plt.ylabel(r'Frac output $\leq1$', fontsize=self.plot_settings.lbl_sz, color=self.plot_settings.lbl_col)
                    plt.xticks(fontsize=self.plot_settings.tk_sz, color=self.plot_settings.tk_col)
                    plt.yticks(fontsize=self.plot_settings.tk_sz, color=self.plot_settings.tk_col)
                plt.legend(loc=self.plot_settings.leg_loc, fontsize=self.plot_settings.leg_sz)
                plt.show()
        
        _plot_stats(10)
        _plot_stats(0)
        _plot_hists()
        _plot_fracs()
        
    def on_train_end(self) -> None:
        self.stats = OrderedDict()
        for h in self.hooks:
            self.stats[h] = self.hooks[h].stats
            self.hooks[h].remove()
        self.plot()
        
        
class MuonConvTelemetryHeadForwards(AbsMuonConvTelemetry):
    '''Draws from from https://github.com/fastai/course-v3/blob/master/nbs/dl2/06_cuda_cnn_hooks_init.ipynb under apache2'''
    
    def __init__(self): super().__init__(hist_range=[-1,20])
        
    @staticmethod
    def hook_fn(hook:ForwardHook, module:nn.Module, input:Tensor, output:Tensor, hist_range:Tuple[float,float]) -> None:
        if not hasattr(hook,'stats'): hook.stats = ([],[],[])
        hook.stats[0].append(float(output[:,1:].data.mean().cpu().detach()))
        hook.stats[1].append(float(output[:,1:].data.std().cpu().detach()))
        hook.stats[2].append(output[:,1:].data.cpu().histc(40,*hist_range))
        
    def on_train_begin(self) -> None:
        self.hooks = OrderedDict()
        for n,m in self.model.head.matrix_head.layers.named_children(): self.hooks[n] = ForwardHook(m, partial(self.hook_fn, hist_range=self.hist_range))
            
            
class MuonConvTelemetryHeadBackwards(AbsMuonConvTelemetry):
    '''Draws from from https://github.com/fastai/course-v3/blob/master/nbs/dl2/06_cuda_cnn_hooks_init.ipynb under apache2'''
    
    def __init__(self): super().__init__(hist_range=[-1,1])
        
    @staticmethod
    def hook_fn(hook:ForwardHook, module:nn.Module, input:Tensor, output:Tensor, hist_range:Tuple[float,float]) -> None:
        if not hasattr(hook,'stats'): hook.stats = ([],[],[])
        hook.stats[0].append(float(output[0,1:].data.mean().cpu().detach()))
        hook.stats[1].append(float(output[0,1:].data.std().cpu().detach()))
        hook.stats[2].append(output[0,1:].data.cpu().histc(40,*hist_range))
        
    def on_train_begin(self) -> None:
        self.hooks = OrderedDict()
        for n,m in self.model.head.matrix_head.layers.named_children(): self.hooks[n] = BackwardHook(m, partial(self.hook_fn, hist_range=self.hist_range))
            

class MuonConvTelemetryBodyForwards(AbsMuonConvTelemetry):
    '''Draws from from https://github.com/fastai/course-v3/blob/master/nbs/dl2/06_cuda_cnn_hooks_init.ipynb under apache2'''
    
    def __init__(self): super().__init__(hist_range=[-1,20])
        
    @staticmethod
    def hook_fn(hook:ForwardHook, module:nn.Module, input:Tensor, output:Tensor, hist_range:Tuple[float,float]) -> None:
        if not hasattr(hook,'stats'): hook.stats = ([],[],[])
        hook.stats[0].append(float(output.data.mean().cpu().detach()))
        hook.stats[1].append(float(output.data.std().cpu().detach()))
        hook.stats[2].append(output.data.cpu().histc(40,*hist_range))
        
    def on_train_begin(self) -> None:
        self.hooks = OrderedDict()
        for n,m in self.model.body.layers.named_children(): self.hooks[n] = ForwardHook(m, partial(self.hook_fn, hist_range=self.hist_range))
            
            
class MuonConvTelemetryBodyBackwards(AbsMuonConvTelemetry):
    '''Draws from from https://github.com/fastai/course-v3/blob/master/nbs/dl2/06_cuda_cnn_hooks_init.ipynb under apache2'''
    
    def __init__(self): super().__init__(hist_range=[-1,1])
        
    @staticmethod
    def hook_fn(hook:ForwardHook, module:nn.Module, input:Tensor, output:Tensor, hist_range:Tuple[float,float]) -> None:
        if not hasattr(hook,'stats'): hook.stats = ([],[],[])
        hook.stats[0].append(float(output[0].data.mean().cpu().detach()))
        hook.stats[1].append(float(output[0].data.std().cpu().detach()))
        hook.stats[2].append(output[0].data.cpu().histc(40,*hist_range))
        
    def on_train_begin(self) -> None:
        self.hooks = OrderedDict()
        for n,m in self.model.body.layers.named_children(): self.hooks[n] = BackwardHook(m, partial(self.hook_fn, hist_range=self.hist_range))
        