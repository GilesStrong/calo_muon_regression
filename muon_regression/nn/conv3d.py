from typing import List, Optional, Any, Callable, Tuple, Union
import numpy as np

import torch
from torch import Tensor
import torch.nn as nn

from lumin.nn.models.initialisations import lookup_normal_init
from lumin.nn.models.layers.activations import lookup_act
from lumin.nn.models.layers.batchnorms import RunningBatchNorm3d
from lumin.utils.misc import to_device, to_tensor
from lumin.nn.models.blocks.conv_blocks import SEBlock1d


__all__ = ['PreActMuonConv3dBlock']


class SEBlock3d(SEBlock1d):
    r'''
    Copyright 2018 onwards Giles Strong

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    Include fix from lumin=0.8.1

    Squeeze-excitation block [Hu, Shen, Albanie, Sun, & Wu, 2017](https://arxiv.org/abs/1709.01507).
    Incoming data is averaged per channel, fed through a single layer of width `n_in//r` and the chose activation, then a second layer of width `n_in` and a sigmoid activation.
    Channels in the original data are then multiplied by the learned channe weights.
    Arguments:
        n_in: number of incoming channels
        r: the reduction ratio for the channel compression
        act: string representation of argument to pass to lookup_act
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
        lookup_act: function taking choice of activation function and returning an activation function layer
    '''

    def __init__(self, n_in:int, r:int, act:str='relu', lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init,
                 lookup_act:Callable[[str],Any]=lookup_act):
        super().__init__(n_in=n_in, r=r)
        self.n_in,self.r,self.act,self.lookup_init,self.lookup_act = n_in,r,act,lookup_init,lookup_act
        self.layers = self._get_layers()
        self.sz = [1,1,1]
        self.pool = nn.AdaptiveAvgPool3d(self.sz)


class PreActMuonConv3dBlock(nn.Module):
    r'''
    Pre-activation Conv3D block with better placement of BN layers and slightly modified shortcut layers. Energy is expected to be channel 0.

    Arguments:
        in_c: number of incoming channels
        conv_out_c: number of outgoing channels for linear path
        e_sum_kernel_sz: kernel size for energy path
        conv_kernel_sz: kernel size for non-linear conv
        pre_act: whether to place teh activation function before the conv layers,instead of after
        grid: (z,x,y) coordinates relevant to the data that will be passed through the model.
            If not `None`, will concatenate the (z,x,y) coordinates of the energy deposits with the energy when computing the shortcut and conv paths.
        mean_subtract_grid: whether to preprocess the grid coordinates via mean subtraction
        act: activation function passed to `get_layers`
        bn: whether to apply batchnormalisation (does not affect choice for use_post_res_bn)
        use_pre_bn: whether to place a low-momentum batchnorm layer at the very start of the network
        running_bn: whether to use running batchnorm or normal batchnorm
        se_net_r: if not `None`, will add squeeze-excitation for channels on the conv path of the residual connections.
            The width of the encoded  representation is then `number of channels // se_net_r`.
        expansion: if > 1, will use 3 conv layers and allow the number of channels to increase inside, before reducing them to the expected output number
        groups: if > 1, will use 3 conv layers, with a grouped convs in the middle convolutional layer. The value of expansion is not considered.
        agg_channel_coef: for ResNeXt style architectures, controls the numer of channels per group according to max(1,n_out_channels*agg_channel_coef)
        lookup_init: function taking choice of activation function, number of inputs, and number of outputs an returning a function to initialise layer weights.
        lookup_act: function taking choice of activation function and returning an activation function layer
    '''

    def __init__(self, in_c:int, conv_out_c:int,
                 e_sum_kernel_sz:Tuple[int,int,int], conv_kernel_sz:Tuple[int,int,int],
                 pre_act:bool, grid:Optional[np.ndarray]=None, mean_subtract_grid:bool=True,
                 act:str='relu', bn:bool=False, use_pre_bn:bool=False, running_bn:bool=False,
                 se_net_r:Optional[int]=None, expansion:int=1, groups:int=1, agg_channel_coef:float=0.25,
                 lookup_init:Callable[[str,Optional[int],Optional[int]],Callable[[Tensor],None]]=lookup_normal_init,
                 lookup_act:Callable[[str],Any]=lookup_act):
        super().__init__()
        self.in_c,self.conv_out_c = in_c,conv_out_c
        self.e_sum_kernel_sz,self.conv_kernel_sz = e_sum_kernel_sz,conv_kernel_sz
        self.pre_act,self.use_pre_bn = pre_act,use_pre_bn
        self.act,self.bn = act,bn
        self.lookup_init,self.lookup_act = lookup_init,lookup_act
        self.se_net_r,self.expansion,self.groups,self.agg_channel_coef = se_net_r,expansion,groups,agg_channel_coef

        self.grid = grid
        if self.grid is not None:
            self.grid = to_tensor(np.expand_dims(self.grid, axis=0))
            if mean_subtract_grid:
                for c in range(self.grid.shape[1]): self.grid[0,c] -= self.grid[0,c].mean()
            self.in_c += 3

        self.bn_layer = RunningBatchNorm3d if running_bn else nn.BatchNorm3d
        self._set_layers()
                
    def _freeze_block(self, block:nn.Module) -> None:
        for p in block.parameters(): p.requires_grad = False
    
    @staticmethod
    def get_padding(kernel_sz:Union[int,Tuple[int]]) -> Union[int,List[int]]:
        r'''
        Automatically computes the required padding to keep the number of columns equal before and after convolution

        Arguments:
            kernel_sz: width of convolutional kernel

        Returns:
            size of padding
        '''

        return kernel_sz//2 if isinstance(kernel_sz, int) else [i//2 for i in kernel_sz]
        
    def _set_layers(self) -> None:
        r'''
        
        '''

        if self.use_pre_bn: self.pre_bn = self.bn_layer(self.in_c)
            
        if self.e_sum_kernel_sz != 1:
            self.e_sum = self.get_conv_layer(in_c=1, out_c=1, kernel_sz=self.e_sum_kernel_sz, stride=self.e_sum_kernel_sz, padding='auto', act=False,
                                             init=nn.init.ones_)
            self._freeze_block(self.e_sum)
        else:
            self.e_sum = lambda x: x

        if self.e_sum_kernel_sz != 1 or self.conv_out_c != self.in_c: self.shortcut = self._get_shortcut_layer()
        else:                                                         self.shortcut = lambda x: x
        
        self.conv_block = self._get_conv_path()
        
    def _get_conv_path(self) -> nn.Sequential:
        if self.expansion == 1 and self.groups == 1:
            layers = [self.get_conv_layer(in_c=self.in_c,       out_c=self.conv_out_c, kernel_sz=self.conv_kernel_sz, bn=self.bn,
                                          stride=self.e_sum_kernel_sz, pre_act=self.pre_act, padding='auto'),
                      self.get_conv_layer(in_c=self.conv_out_c, out_c=self.conv_out_c, kernel_sz=3,                   bn=self.bn,
                                          stride=1,                    pre_act=self.pre_act)]
        else:
            if self.groups > 1:  # RexNeXt
                inner_c_1 = np.max((1,int(self.conv_out_c*self.agg_channel_coef)))*self.groups
                inner_c_2 = inner_c_1
            else:  # ResNet50
                inner_c_1 = self.conv_out_c*2
                inner_c_2 = int(self.conv_out_c*self.expansion)
            layers = [self.get_conv_layer(in_c=self.in_c, out_c=inner_c_1,         kernel_sz=1,                   bn=self.bn,
                                          stride=1,                    pre_act=self.pre_act),
                      self.get_conv_layer(in_c=inner_c_1, out_c=inner_c_2,         kernel_sz=self.conv_kernel_sz, bn=self.bn,
                                          stride=self.e_sum_kernel_sz, pre_act=self.pre_act, padding='auto', groups=self.groups),
                      self.get_conv_layer(in_c=inner_c_2, out_c=self.conv_out_c,   kernel_sz=1,                   bn=self.bn,
                                          stride=1,                    pre_act=self.pre_act)]
        if self.se_net_r is not None: layers.append(SEBlock3d(self.conv_out_c, self.se_net_r, act=self.act, lookup_init=self.lookup_init,
                                                              lookup_act=self.lookup_act))
        return nn.Sequential(*layers)
            
    def _get_shortcut_layer(self) -> nn.Sequential:
        shortcut = []
        if self.pre_act: shortcut += [self.lookup_act(self.act), self.bn_layer(self.in_c)]
        shortcut += [nn.AvgPool3d(self.e_sum_kernel_sz, stride=self.e_sum_kernel_sz, padding=self.get_padding(self.e_sum_kernel_sz)),
                     nn.Conv3d(in_channels=self.in_c, out_channels=self.conv_out_c, kernel_size=1, stride=1, bias=not self.pre_act)]
        self.lookup_init(self.act)(shortcut[-1].weight)
        return nn.Sequential(*shortcut)

    def get_conv_layer(self, in_c:int, out_c:int, kernel_sz:int, padding:Union[int,Tuple[int],str]='auto', stride:int=1, pre_act:bool=False, groups:int=1,
                       bias:bool=False, act:bool=True, bn:bool=False, init:Union[str,Callable]='auto') -> nn.Module:
        r'''
        Builds a sandwich of layers with a single convolutional layer, plus any requested batch norm and activation.
        Also initialises layers to requested scheme.

        Arguments:
            in_c: number of input channels (number of features per object / rows in input matrix)
            conv_out_c: number of output channels (number of features / rows in output matrix)
            kernel_sz: width of kernel, i.e. the number of columns to overlay
            padding: amount of padding columns to add at start and end of convolution.
                If left as 'auto', padding will be automatically computed to conserve the number of columns.
            stride: number of columns to move kernel when computing convolutions. Stride 1 = kernel centred on each column,
                stride 2 = kernel centred on ever other column and input size halved, et cetera.
            pre_act: whether to apply batchnorm and activation layers prior to the weight layer, or afterwards
            groups: number of blocks of connections from input channels to output channels
            bias: whether to have bias parameters
            act: whether to add and activation function
            bn: whether to use batchnormalisation
            init: initialisation scheme to use for weights
        '''
        
        if padding == 'auto': padding = self.get_padding(kernel_sz)
        layers = []
        if pre_act:
            if bn: layers.append(self.bn_layer(in_c))
            if self.act != 'linear' and act: layers.append(self.lookup_act(self.act))
                    
        layers.append(nn.Conv3d(in_channels=in_c, out_channels=out_c, kernel_size=kernel_sz, padding=padding, stride=stride, groups=groups, bias=bias))
        if init == 'auto': self.lookup_init(self.act if act else 'linear')(layers[-1].weight)
        else:              init(layers[-1].weight)
        if bias: nn.init.zeros_(layers[-1].bias)
        
        if not pre_act:
            if self.act != 'linear' and act: layers.append(self.lookup_act(self.act))
            if bn: layers.append(self.bn_layer(out_c))
        return nn.Sequential(*layers)

    def forward(self, x:Tensor, debug:bool=False) -> Tensor:
        r'''
        Passes input through the layers.

        Arguments:
            x: input tensor
        
        Returns:
            Resulting tensor
        '''
        if self.use_pre_bn: x = self.pre_bn(x)
#         print("\npre", x.shape)
        x_e_sum = self.e_sum(x[:,:1])  # Channel 0 is energy
#         print("x_e_sum", x_e_sum.shape)
        if self.grid is not None:
            if self.grid.device != x.device: self.grid = to_device(self.grid, torch.device(x.device))
            x = torch.cat((x, self.grid.expand((x.shape[0],*self.grid.shape[1:]))), 1)  # Expand and concat x,y,z
        x_conv     = self.conv_block(x)
#         print("x_conv", x_conv.shape)
        x_shortcut = self.shortcut(x)
#         print("x_shortcut", x_shortcut.shape)
        x_conv     = x_shortcut+x_conv
#         print("x_post_res", x_conv.shape, '\n')
        return torch.cat((x_e_sum, x_conv), 1)
