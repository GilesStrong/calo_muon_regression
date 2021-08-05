import numpy as np  # noqa F401
import pandas as pd  # noqa F401
import math  # noqa F401
import os  # noqa F401
import types  # noqa F401
import h5py  # noqa F401
import pickle  # noqa F401
from pathlib import Path  # noqa F401
from typing import *  # noqa F403 F401
import multiprocessing as mp  # noqa F401
import json  # noqa F401
from functools import partial  # noqa F401
import timeit  # noqa F401
from fastprogress import progress_bar, master_bar  # noqa F401

import torch.nn as nn  # noqa F401
from torch.tensor import Tensor  # noqa F401
import torch  # noqa F401

from lumin.nn.training.train import *  # noqa F403 F401
from lumin.nn.models.model_builder import ModelBuilder  # noqa F401
from lumin.nn.models.model import Model  # noqa F401
from lumin.nn.data.fold_yielder import *  # noqa F403 F401
from lumin.nn.ensemble.ensemble import Ensemble  # noqa F401
from lumin.nn.metrics.class_eval import *  # noqa F403 F401
from lumin.plotting.data_viewing import *  # noqa F403 F401
from lumin.utils.misc import *  # noqa F403 F401
from lumin.optimisation.threshold import binary_class_cut_by_ams  # noqa F401
from lumin.optimisation.hyper_param import lr_find  # noqa F401
from lumin.evaluation.ams import calc_ams  # noqa F401
from lumin.nn.callbacks.cyclic_callbacks import CycleStep as StepCycle  # noqa F401
from lumin.nn.callbacks.model_callbacks import *  # noqa F403 F401
from lumin.nn.callbacks.data_callbacks import *  # noqa F403 F401
from lumin.nn.callbacks.loss_callbacks import *  # noqa F403 F401
from lumin.nn.callbacks.monitors import *  # noqa F403 F401
from lumin.nn.callbacks.lsuv_init import LsuvInit  # noqa F401
from lumin.plotting.results import *  # noqa F403 F401
from lumin.plotting.training import *  # noqa F403 F401
from lumin.plotting.plot_settings import PlotSettings  # noqa F401
from lumin.plotting.interpretation import *  # noqa F403 F401
from lumin.nn.losses.basic_weighted import *  # noqa F403 F401
from lumin.nn.losses.advanced_losses import * # noqa F403 F401
from lumin.nn.models.helpers import CatEmbedder  # noqa F401
from lumin.nn.models.blocks.head import *  # noqa F403 F401
from lumin.nn.models.blocks.body import *  # noqa F403 F401
from lumin.nn.models.blocks.tail import *  # noqa F403 F401
from lumin.utils.statistics import uncert_round  # noqa F401
from lumin.data_processing.file_proc import df2foldfile, fold2foldfile, add_meta_data  # noqa F401
from lumin.nn.models.blocks.gnn_blocks import GravNet, GraphCollapser  # noqa F401

from .experiment import *  # noqa F304
from .data_proc import  *  # noqa F304
from .nn import  *  # noqa F304
from .plotting import *  # noqa F304
from .result_proc import *   # noqa F304
from .fqs import *   # noqa F304

import seaborn as sns
import matplotlib.pyplot as plt  # noqa F401
sns.set_style("whitegrid")

DATA_PATH    = Path("../data/")
RESULTS_PATH = Path("../results/")
plot_settings = PlotSettings(cat_palette='tab10', savepath=Path('plots'), format='.pdf')

ignore_feats = [
    'eabove',
    'ncellmax',
    'eclumax',
    'nclu1',
    'ncellmax1',
    'eclumax1',
    'epercell1',
    'maxsum',
    'ea1'
]

esum_only_feats = [
    'eabove',
    'eb1',
    'ea1'
]
