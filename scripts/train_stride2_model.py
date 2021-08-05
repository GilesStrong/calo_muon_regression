import sys
from typer import Typer, Argument, Option
from functools import partial
import numpy as np
import pandas as pd
from pathlib import Path
import timeit
from typing import Optional, List, Dict, Tuple, Union

import torch

from lumin.nn.data.fold_yielder import FoldYielder
from lumin.nn.models.model_builder import ModelBuilder
from lumin.nn.models.model import Model
from lumin.nn.models.blocks.head import MultiHead, CatEmbHead
from lumin.nn.models.blocks.body import FullyConnected
from lumin.nn.models.blocks.tail import ClassRegMulti
from lumin.nn.ensemble.ensemble import Ensemble
from lumin.nn.training.train import train_models
from lumin.optimisation.hyper_param import lr_find
from lumin.plotting.training import plot_train_history
from lumin.nn.callbacks.monitors import EpochSaver
from lumin.nn.callbacks.cyclic_callbacks import CycleStep
from lumin.nn.losses.advanced_losses import WeightedFractionalBinnedHuber

sys.path.append('../')
from muon_regression.nn import Stride2PreActMuonConvTensorHead  # noqa E402
from muon_regression.nn import DownweightHighEMuons, MaxRMSE, ImprovRMSE  # noqa E402
from muon_regression.experiment import ExpJournal  # noqa E402
from muon_regression.basics import plot_settings, RESULTS_PATH, ignore_feats, esum_only_feats  # noqa E402
from muon_regression.plotting import produce_plots , PowerFunc, LinFunc # noqa E402

app = Typer()


def get_model_builder(train_fy:FoldYielder,
                      n_blocks:int=3,
                      n_fc_layers:int=3,
                      fc_layer_width:int=80,
                      first_kernel_sz:int=4,
                      act:str='swish',
                      use_bn:bool=True,
                      use_fc_bn:bool=False,
                      use_pre_bn:bool=False,
                      grid:Optional[np.ndarray]=None,
                      wd:float=0,
                      do:float=0,
                      n_full_res:int=0,
                      pooling_sz:Optional[Union[Tuple[int,int,int],int]]=(1,1,1),
                      c_coef:float=1.5,
                      running_bn:bool=True,
                      se_net_r:int=4,
                      groups:int=1,
                      jrn:Optional[ExpJournal]=None) -> ModelBuilder:
    if train_fy.has_matrix:
        conv_head  = partial(Stride2PreActMuonConvTensorHead,
                             shape=train_fy.matrix_feats['shape'],
                             act=act,
                             bn=use_bn,
                             layer_kargs={'n_blocks_per_res':n_blocks,
                                          'first_kernel_sz':first_kernel_sz,
                                          'use_pre_bn':use_pre_bn,
                                          'grid':grid,
                                          'n_full_res':n_full_res,
                                          'pooling_sz':pooling_sz,
                                          'c_coef':c_coef,
                                          'running_bn':running_bn,
                                          'se_net_r':se_net_r,
                                          'groups':groups},
                             vecs=['rechit'],
                             feats_per_vec=['energy'])
        if len(train_fy.cont_feats) > 0:
            head = partial(MultiHead, matrix_head=conv_head)
            cont_feats = train_fy.cont_feats+train_fy.matrix_feats['present_feats']
        else:
            head = conv_head
            cont_feats = train_fy.matrix_feats['present_feats']
    else:
        head = CatEmbHead
        cont_feats = train_fy.cont_feats
    model_builder = ModelBuilder(objective='regressor',
                                 cont_feats=cont_feats,
                                 n_out=1,
                                 opt_args={'opt':'adamw', 'eps':1e-08, 'weight_decay':wd} if wd > 0 else {'opt':'adam', 'eps':1e-08},
                                 head=head,
                                 body=partial(FullyConnected, depth=n_fc_layers, width=fc_layer_width, act=act, bn=use_fc_bn, do=do),
                                 tail=partial(ClassRegMulti),
                                 loss=partial(WeightedFractionalBinnedHuber, bins=torch.linspace(50,8000,6), perc=0.68))
    m = Model(model_builder)
    print("Model configuration", m,'\n')
    if jrn is not None:
        jrn['model_size'] = m.get_param_count()
        jrn['model_params'] = {'n_blocks':n_blocks,
                               'n_fc_layers':n_fc_layers,
                               'fc_layer_width':fc_layer_width,
                               'first_kernel_sz':first_kernel_sz,
                               'act':act,
                               'use_bn':use_bn,
                               'use_fc_bn':use_fc_bn,
                               'use_pre_bn':use_pre_bn,
                               'grid':grid,
                               'wd':wd,
                               'do':do, 
                               'n_full_res':n_full_res,
                               'pooling_sz':pooling_sz, 
                               'c_coef':c_coef, 
                               'running_bn':running_bn,
                               'se_net_r':None if se_net_r == 0 else se_net_r,
                               'groups':groups}
    return model_builder


def train_model(model_builder:ModelBuilder,
                train_fy:FoldYielder,
                n_epochs:int,
                lr:float,
                bs:int,
                jrn:ExpJournal,
                cycle_warmup_len:int=2,
                cycle_decay_len:int=18,
                cycle_final_lr_frac:float=10,
                tmp_train_path:Path=Path('train_weights'),
                epoch_save:bool=False,
                n_models:int=1,
                excl_idxs:Optional[List[int]]=None,
                unique_trn_idxs:bool=False) -> Tuple[List[Dict[str,float]],List[Dict[str,List[float]]]]:
    callback_partials = [DownweightHighEMuons,
                         partial(CycleStep,
                                 frac_reduction=0.5,
                                 patience=2,
                                 lengths=(cycle_warmup_len, cycle_decay_len),
                                 lr_range=[lr/100, lr, lr/cycle_final_lr_frac],
                                 mom_range=(0.85, 0.95),
                                 interp='cosine')]
    if epoch_save: callback_partials.append(EpochSaver)
    metric_partials = [partial(ImprovRMSE, n_bins=20, partial_fit_func=LinFunc,   bootstrap_n=None, restricted_energy=True,  main_metric=False),
                       partial(MaxRMSE,    n_bins=20, partial_fit_func=LinFunc,   bootstrap_n=None, restricted_energy=True,  main_metric=False),
                       partial(ImprovRMSE, n_bins=20, partial_fit_func=PowerFunc, bootstrap_n=None, restricted_energy=False, main_metric=False),
                       partial(MaxRMSE,    n_bins=20, partial_fit_func=PowerFunc, bootstrap_n=None, restricted_energy=False, main_metric=False)]
    print(f'Training {n_models} model(s) for upto {n_epochs} epochs at a batch size of {bs}. \
            StepCycle warm up for {cycle_warmup_len} epochs and decay for {cycle_decay_len}.')
    tmr = timeit.default_timer()
    results,histories,_ = train_models(fy=train_fy,
                                       n_models=n_models,
                                       model_builder=model_builder,
                                       bs=bs,
                                       cb_partials=callback_partials,
                                       n_epochs=n_epochs,
                                       patience=10,
                                       metric_partials=metric_partials,
                                       savepath=tmp_train_path,
                                       bulk_move=False,
                                       excl_idxs=excl_idxs,
                                       unique_trn_idxs=unique_trn_idxs)
    jrn['train_time'] = timeit.default_timer()-tmr
    return results, histories


def predict_data(ensemble:Ensemble, train_fy:FoldYielder, jrn:ExpJournal, bs:int, fold_idx:int) -> pd.DataFrame:
    tmr = timeit.default_timer()
    val = train_fy[fold_idx]
    pred = ensemble.predict(val['inputs'], bs=bs)
    df = pd.DataFrame({'gen_target':val['targets'].squeeze(), 'pred':pred.squeeze()})
    jrn['val_time'] = timeit.default_timer()-tmr
    return df


def record_mean_results(results:List[Dict[str,float]], jrn:ExpJournal) -> None:
    for score in results[0]:
        if score == 'path': continue
        jrn[f'mean_{score}'] = [float(np.mean([x[score] for x in results])),float(np.std([x[score] for x in results])/np.sqrt(len(results)))]

    
@app.command()
def lr_range_test(run_name:      str=Argument(...,          help='Run name for the training, used for saving the results.'),
                  file_name:     str=Argument(...,          help='Name of foldfile containing data for training and validation.'),
                  results_path:  str=Argument(RESULTS_PATH, help='Directory path in which to save results.'),
                  bs:             int=Option(256,   help='Batch size',                                         min=1),
                  n_blocks:       int=Option(3,     help='Number of blocks to use at each grid resolution.'),
                  n_runs:         int=Option(1,     help='Number of times to run the LR range test.',          min=1, max=10),
                  n_fc_layers:    int=Option(3,     help='Fully connected layers to use in the network body.', min=1),
                  fc_layer_width: int=Option(80,    help='Width of FC layers in network body.',                min=1),
                  wd:           float=Option(0,     help='Weight-decay coeficient (non-zero = AdamW optimiser).'),
                  do:           float=Option(0,     help='Dropout rate for fully-connected layers.'),
                  n_full_res:     int=Option(0,     help='Number of block at full resolution.'),
                  gpu_id:         int=Option(0,     help='GPU ID for CUDA device'),
                  c_coef:       float=Option(1.5,   help="Expansion rate for channels during downsampling"),
                  se_net_r:       int=Option(4,     help='Compression factor for SE block, set to zero for no SE'),
                  groups:         int=Option(1,     help='If > 1, will use a ResNeXt style architecture with aggregated residual connections \
                                                          with cardinality euqual to argument'),
                  hl_feat_mode:   str=Option('default', help='Controls which HL-features are used, if present in data: \
                                                              default=use all features, \
                                                              filtered=ignore pre-specified features, \
                                                              esum=only use reco-energy features.'),
                  pooling_sz:Tuple[int,int,int]=Option((1,1,1), help='Size of adaptive average pooling in z,x,y. \
                                                                      Set to (0,0,0) for no adaptive pooling')) -> None:
    
    r'''
    Example: python train_stride2_model.py lr-range-test test_1 ../data/xmas_calo_hl_small_5.hdf5 padova_v100
    '''

    # Prepare
    torch.cuda.set_device(gpu_id)
    results_path = Path(results_path)/run_name
    results_path.mkdir(parents=True, exist_ok=True)
    plot_settings.savepath = results_path/'plots'
    plot_settings.savepath.mkdir(parents=True, exist_ok=True)
    if pooling_sz[0] is None or pooling_sz[0] == 0: pooling_sz = None

    # Training
    train_fy = FoldYielder(file_name)
    if   hl_feat_mode == 'filtered': train_fy.add_ignore(ignore_feats)
    elif hl_feat_mode == 'esum':     train_fy.add_ignore([f for f in train_fy.cont_feats if f not in esum_only_feats])
    
    model_builder = get_model_builder(train_fy,
                                      n_blocks=n_blocks,
                                      n_fc_layers=n_fc_layers,
                                      fc_layer_width=fc_layer_width,
                                      wd=wd,
                                      do=do,
                                      n_full_res=n_full_res,
                                      pooling_sz=pooling_sz,
                                      c_coef=c_coef,
                                      se_net_r=None if se_net_r == 0 else se_net_r,
                                      groups=groups)
    print("Running LR range test")
    lr_find(fy=train_fy,
            model_builder=model_builder,
            bs=bs,
            lr_bounds=[1e-7,1e1],
            n_folds=n_runs,
            plot_settings=plot_settings,
            plot_savename='lr_rangetest')    
    print('\nTest complete, plot saved to:', plot_settings.savepath/f'lr_rangetest{plot_settings.format}')


@app.command()
def train(run_name:             str=Argument(...,             help='Run name for the training, used for saving the results.'),
          file_name:            str=Argument(...,             help='Name of foldfile containing data for training and validation.'),
          machine_name:         str=Argument('unnamed',       help='Name of current machine (as hardcoded in ExpJournal class)'),
          results_path:         str=Argument(RESULTS_PATH,    help='Directory path in which to save results (will create a new directory called file_name'),
          n_epochs:             int=Option(50,    help='Number of epochs for which to train.',               min=1),
          lr:                 float=Option(5e-3,  help='Nominal learning rate'),
          bs:                   int=Option(256,   help='Batch size',                                         min=1),
          n_blocks:             int=Option(3,     help='Number of blocks to use at each grid resolution.'),
          n_fc_layers:          int=Option(3,     help='Fully connected layers to use in the network body.', min=1),
          fc_layer_width:       int=Option(80,    help='Width of FC layers in network body.',                min=1),
          wd:                 float=Option(0,     help='Weight-decay coefficient (non-zero = AdamW optimiser).'),
          do:                 float=Option(0,     help='Dropout rate for fully-connected layers.'),
          n_full_res:           int=Option(0,     help='Number of block at full resolution.'),
          gpu_id:               int=Option(0,     help='GPU ID for CUDA device'),
          epoch_save:          bool=Option(False, help="Whether to save the model after every epoch"),
          c_coef:             float=Option(1.5,   help="Expansion rate for channels during downsampling"),
          se_net_r:             int=Option(4,     help='Compression factor for SE block, set to zero for no SE'),
          cycle_warmup_len:     int=Option(2,     help='Number of warmup epochs for OneCycle part of StepCycle'),
          cycle_decay_len:      int=Option(18,    help='Number of decay epochs for OneCycle part of StepCycle'),
          cycle_final_lr_frac:float=Option(10,    help='Fractional reduction of LR for endpoint of OneCycle part of StepCycle'),
          groups:               int=Option(1,     help='If > 1, will use a ResNeXt style architecture with aggregated residual connections \
                                                        with cardinality equal to argument'),
          n_models:             int=Option(1,     help='Number of models to train. Average training metrics stats will be recorded, \
                                                        but only the best model will be evaluated further', min=1),
          extra_val_data:      bool=Option(False, help='If true, will exclude fold zero form training and monitoring-validation and \
                                                        use it to evaluate final performance. \
                                                        Will also evaulate entire ensemble, rather than just best performing model'),
          unique_trn_idxs:     bool=Option(False, help='If True, will split all folds into groups of unique folds and train each model on a single group,\
                                                        i.e no fold is used to train more than one model.'),
          hl_feat_mode:         str=Option('default', help='Controls which HL-features are used, if present in data: \
                                                            default=use all features, \
                                                            filtered=ignore pre-specified features, \
                                                            esum=only use reco-energy features.'),
          pooling_sz:Tuple[int,int,int]=Option((1,1,1), help='Size of adaptive average pooling in z,x,y. Set to (0,0,0) for no adaptive pooling')) -> None:

    r'''
    Example: python train_stride2_model.py train --n-epochs=5  --cycle-decay-len=3 test_1 ../data/xmas_calo_hl_small_5.hdf5 padova_v100
    '''

    # Prepare
    torch.cuda.set_device(gpu_id)
    results_path = Path(results_path)/run_name
    results_path.mkdir(parents=True, exist_ok=True)
    jrn = ExpJournal(exp_name=run_name, machine=machine_name, path=results_path)
    plot_settings.savepath = results_path/'plots'
    plot_settings.savepath.mkdir(parents=True, exist_ok=True)
    if pooling_sz[0] is None or pooling_sz[0] == 0: pooling_sz = None
    
    # Training
    train_fy = FoldYielder(file_name)
    if   hl_feat_mode == 'filtered': train_fy.add_ignore(ignore_feats)
    elif hl_feat_mode == 'esum':     train_fy.add_ignore([f for f in train_fy.cont_feats if f not in esum_only_feats])
    
    model_builder = get_model_builder(train_fy,
                                      n_blocks=n_blocks,
                                      n_fc_layers=n_fc_layers,
                                      fc_layer_width=fc_layer_width,
                                      wd=wd,
                                      do=do,
                                      n_full_res=n_full_res,
                                      pooling_sz=pooling_sz,
                                      c_coef=c_coef,
                                      se_net_r=None if se_net_r == 0 else se_net_r,
                                      groups=groups,
                                      jrn=jrn)
    results, histories = train_model(model_builder,
                                     train_fy,
                                     n_epochs=n_epochs,
                                     lr=lr,
                                     bs=bs,
                                     jrn=jrn,
                                     tmp_train_path=results_path/'train_weights',
                                     epoch_save=epoch_save,
                                     cycle_warmup_len=cycle_warmup_len,
                                     cycle_decay_len=cycle_decay_len,
                                     cycle_final_lr_frac=cycle_final_lr_frac,
                                     n_models=n_models,
                                     excl_idxs=[0] if extra_val_data else None,
                                     unique_trn_idxs=unique_trn_idxs)
    plot_train_history(histories, 'Full_loss_history', ignore_trn=False, settings=plot_settings, show=False)
    
    # Evaluation
    print('\nTraining complete, loading model')
    record_mean_results(results, jrn)
    ensemble = Ensemble.from_results(results, size=n_models if extra_val_data else 1, model_builder=model_builder)
    ensemble.save(results_path/run_name, overwrite=True)
    print(f'Model saved to weights/{results_path/run_name}')
    print('Evaluating model performance')
    val_fold = 0 if extra_val_data else np.argmin([r['loss'] for r in ensemble.results])
    df = predict_data(ensemble, train_fy, jrn, bs=bs, fold_idx=val_fold)
    print(f'Saving validation dataframe to {results_path}')
    df[[c for c in df.columns if c != 'gen_weight']].to_csv(results_path/'val_df.csv', index=False)
    print(f'\nProducing plots and saving to {plot_settings.savepath.absolute()}')
    produce_plots(df, jrn, plot_settings, fit_func=PowerFunc(), restrict_range=False)
    produce_plots(df, jrn, plot_settings, fit_func=LinFunc(),   restrict_range=True)
    print('\nModel performance:\n', jrn)

    # Closing
    print(f'Run complete, saving results journal to {jrn.path.absolute()}')
    jrn.save()


if __name__ == "__main__": app()
