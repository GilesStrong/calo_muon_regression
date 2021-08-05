from functools import partial
from typing import List, Optional, Dict, Any, Callable, Tuple, Union
from abc import abstractmethod

import torch
from torch import Tensor, nn

from lumin.nn.models.blocks.head import AbsMatrixHead
from lumin.nn.models.blocks.conv_blocks import AdaptiveAvgMaxConcatPool3d
from lumin.nn.models.initialisations import lookup_normal_init
from lumin.nn.models.layers.activations import lookup_act


from .conv3d import PreActMuonConv3dBlock


__all__ = ['Stride2MuonConvTensorHead', 'Stride2PreActMuonConvTensorHead']


class AbsConvTensorHead(AbsMatrixHead):
    r'''
    Abstract wrapper head for applying 3D convolutions to pre-shaped tensor data.
    Users should inherit from this class and overload `get_layers` to define their model.
    
    Arguments:
        cont_feats: list of all the tensor features which are present in the input data
        shape: shape of tensor to expect (exclude data, dimension (i.e. the first dimension))
        basic_block: basic building block for the convolution head
        act: activation function passed to `get_layers`
        bn: batch normalisation argument passed to `get_layers`
        layer_kargs: dictionary of keyword arguments which are passed to `get_layers`
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
        freeze: whether to start with module parameters set to untrainable
    '''

    def __init__(self, cont_feats:List[str], shape:Tuple[int], basic_block:nn.Module, act:str='relu', bn:bool=False, layer_kargs:Optional[Dict[str,Any]]=None,
                 lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init,
                 lookup_act:Callable[[str],Any]=lookup_act, freeze:bool=False, **kargs):
        super().__init__(cont_feats=cont_feats, vecs=[], feats_per_vec=[], lookup_init=lookup_init, lookup_act=lookup_act, freeze=freeze)
        self.shape,self.basic_block = shape,basic_block
        if layer_kargs is None: layer_kargs = {}
        self.layers:nn.Module = self.get_layers(in_c=self.shape[0], act=act, bn=bn, **layer_kargs)
        self.out_sz = self.check_out_sz()
        if self.freeze: self.freeze_layers()
        self._map_outputs()
            
    def _map_outputs(self) -> None:
        self.feat_map = {}
        for i, f in enumerate(self.cont_feats): self.feat_map[f] = list(range(self.get_out_size()))
            
    def check_out_sz(self) -> int:
        r'''
        Automatically computes the output size of the head by passing through random data of the expected shape

        Returns:
            x.size(-1) where x is the outgoing tensor from the head
        '''

        x = torch.zeros((1, *self.shape))
        training = self.training
        self.eval()
        x = self.forward(x)
        if training: self.train()
        return x.size(-1)
    
    @abstractmethod
    def get_layers(self, in_c:int, act:str='relu', bn:bool=False, **kargs) -> nn.Module:
        r'''
        Abstract function to be overloaded by user. Should return a single torch.nn.Module which accepts the expected input matrix data.
        
        '''
        
        # layers = []
        # layers.append(self.get_conv1d_block(in_c, 16, kernel_sz=7, padding=3, stride=2))
        # ...
        # layers = nn.Sequential(*layers)
        # return layers
        
        pass

    def forward(self, x:Union[Tensor,Tuple[Tensor,Tensor]]) -> Tensor:
        r'''
        Passes input through the convolutional network.

        Arguments:
            x: If a tuple, the second element is assumed to the be the matrix data. If a flat tensor, will convert the data to a matrix
        
        Returns:
            Resulting tensor
        '''

        x = self._process_input(x)
        return self.layers(x).view(x.size(0),-1)
    
    def get_out_size(self) -> int:
        r'''
        Get size of output

        Returns:
            Width of output representation
        '''
        
        return self.out_sz


class Stride2MuonConvTensorHead(AbsConvTensorHead):
    def get_layers(self, in_c:int, act:str='relu', bn:bool=False, **kargs) -> nn.Module:
        conv_block = partial(self.basic_block,
                             act=act,
                             bn=bn,
                             lookup_init=self.lookup_init,
                             lookup_act=self.lookup_act,
                             running_bn=kargs['running_bn'] if 'running_bn' in kargs else False,
                             se_net_r=kargs['se_net_r'] if 'se_net_r' in kargs else None,
                             expansion=kargs['expansion'] if 'expansion' in kargs else 1,
                             groups=kargs['groups'] if 'groups' in kargs else 1,
                             agg_channel_coef=kargs['agg_channel_coef'] if 'agg_channel_coef' in kargs else 0.25)
        downsample = partial(conv_block, e_sum_kernel_sz=2, conv_kernel_sz=4, pre_act=True)
        subsequent = partial(conv_block, e_sum_kernel_sz=1, conv_kernel_sz=3, pre_act=True)
        
        layers = []
        n_c = 0
        c_coef = 2 if 'c_coef' not in kargs else kargs['c_coef']
        
        # Full res 50, 32, 32
        if 'n_full_res' in kargs and kargs['n_full_res'] > 0:
            layers.append(conv_block(in_c=in_c,
                                     grid=kargs['grid'] if 'grid' in kargs else None,
                                     mean_subtract_grid=kargs['mean_subtract_grid'] if 'mean_subtract_grid' in kargs else True,
                                     use_pre_bn=kargs['use_pre_bn'] if 'use_pre_bn' in kargs else False,
                                     pre_act=False,
                                     conv_out_c=4,
                                     e_sum_kernel_sz=1,
                                     conv_kernel_sz=kargs['first_kernel_sz'] if 'first_kernel_sz' in kargs else 4))
            n_c += 5
            for i in range(kargs['n_full_res']-1):
                layers.append(conv_block(in_c=n_c,
                                         conv_out_c=n_c,
                                         e_sum_kernel_sz=1,
                                         conv_kernel_sz=kargs['first_kernel_sz'] if 'first_kernel_sz' in kargs else 3,
                                         pre_act=True))
                n_c += 1

        # Downsample 50, 32, 32 -> 26, 17, 17
        if len(layers) == 0:
            layers.append(conv_block(in_c=in_c,
                                     grid=kargs['grid'] if 'grid' in kargs else None,
                                     mean_subtract_grid=kargs['mean_subtract_grid'] if 'mean_subtract_grid' in kargs else True,
                                     use_pre_bn=kargs['use_pre_bn'] if 'use_pre_bn' in kargs else False,
                                     pre_act=False,
                                     conv_out_c=8,
                                     e_sum_kernel_sz=2,
                                     conv_kernel_sz=kargs['first_kernel_sz'] if 'first_kernel_sz' in kargs else 4))
            n_c = 9
        else:
            n_o = int(n_c*c_coef)
            layers.append(conv_block(in_c=n_c,
                                     pre_act=True,
                                     conv_out_c=n_o,
                                     e_sum_kernel_sz=2,
                                     conv_kernel_sz=kargs['first_kernel_sz'] if 'first_kernel_sz' in kargs else 4))
            n_c = n_o+1
                             
        # Subsequent
        if 'n_blocks_per_res' in kargs and kargs['n_blocks_per_res'] > 1:
            for i in range(kargs['n_blocks_per_res']-1): 
                layers.append(subsequent(in_c=n_c, conv_out_c=n_c))
                n_c += 1

        # Downsample 26, 17, 17 -> 14, 9, 9
        n_o = int(n_c*c_coef)
        layers.append(downsample(in_c=n_c, conv_out_c=n_o))
        n_c = n_o+1
        
        # Subsequent    
        if 'n_blocks_per_res' in kargs and kargs['n_blocks_per_res'] > 1:
            for i in range(kargs['n_blocks_per_res']-1): 
                layers.append(subsequent(in_c=n_c, conv_out_c=n_c))
                n_c += 1
            
        # Downsample 14, 9, 9 -> 8, 5, 5
        n_o = int(n_c*c_coef)
        layers.append(downsample(in_c=n_c, conv_out_c=n_o))
        n_c = n_o+1
        
        # Subsequent    
        if 'n_blocks_per_res' in kargs and kargs['n_blocks_per_res'] > 1:
            for i in range(kargs['n_blocks_per_res']-1): 
                layers.append(subsequent(in_c=n_c, conv_out_c=n_c))
                n_c += 1
                             
        # Downsample 8, 5, 5 -> 5, 3, 3
        n_o = int(n_c*c_coef)
        layers.append(downsample(in_c=n_c, conv_out_c=n_o))
        n_c = n_o+1
        
        # Subsequent    
        if 'n_blocks_per_res' in kargs and kargs['n_blocks_per_res'] > 1:
            for i in range(kargs['n_blocks_per_res']-1): 
                layers.append(subsequent(in_c=n_c, conv_out_c=n_c))
                n_c += 1
        
        if 'pooling_sz' in kargs and kargs['pooling_sz'] is not None: layers.append(AdaptiveAvgMaxConcatPool3d(kargs['pooling_sz']))
        layers = nn.Sequential(*layers)
        return layers


class Stride2PreActMuonConvTensorHead(Stride2MuonConvTensorHead):
    r'''
    Compatibilty wrapper for old code using PreActMuonConv3dBlock
    
    Arguments:
        cont_feats: list of all the tensor features which are present in the input data
        shape: shape of tensor to expect (exclude data, dimension (i.e. the first dimension))
        basic_block: basic building block for the convolution head
        act: activation function passed to `get_layers`
        bn: batch normalisation argument passed to `get_layers`
        layer_kargs: dictionary of keyword arguments which are passed to `get_layers`
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
        freeze: whether to start with module parameters set to untrainable
    '''

    def __init__(self, cont_feats:List[str], shape:Tuple[int], act:str='relu', bn:bool=False, layer_kargs:Optional[Dict[str,Any]]=None,
                 lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init,
                 lookup_act:Callable[[str],Any]=lookup_act, freeze:bool=False, **kargs):
        super().__init__(cont_feats=cont_feats, shape=shape, basic_block=PreActMuonConv3dBlock, act=act, bn=bn, layer_kargs=layer_kargs, lookup_init=lookup_init, lookup_act=lookup_act, freeze=freeze, **kargs)
        