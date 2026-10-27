from __future__ import annotations
import argparse, copy, logging, math, os, random
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import h5py as h5
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from diffusers import DDIMScheduler
from tqdm import tqdm
from copy import deepcopy
_ACTIVATION_FUNCTIONS = {
    'relu': nn.ReLU(),
    # 'prelu': nn.PReLU(),
    'rrlu': nn.RReLU(),
    'relu6': nn.ReLU6(),
    'leaky_relu': nn.LeakyReLU(),
    'elu': nn.ELU(),
    'selu': nn.SELU(),
    'celu': nn.CELU(),
    'gelu': nn.GELU(),
    'silu': nn.SiLU(),
    'mish': nn.Mish(),
    'tanh': nn.Tanh(),
    'hardtanh': nn.Hardtanh(),
    'sigmoid': nn.Sigmoid(),
    'log_sigmoid': nn.LogSigmoid(),
    'softplus': nn.Softplus(),
    'softsign': nn.Softsign(),
    'hardshrink': nn.Hardshrink(),
    'tanhshrink': nn.Tanhshrink(),
    'softshrink': nn.Softshrink(),
}

class IdentityMerger(nn.Module):
    legal_nargs = 1
    def __init__(self):
        super(IdentityMerger, self).__init__()

    def forward(self, x_list):
        return x_list[0]

class BasePrimitive(nn.Module):
    """
    Base class for all primitives.
    """
    legal_ndims = []

    def __init__(self, *args, **kwargs):
        super(BasePrimitive, self).__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x):
        raise NotImplementedError("Subclasses should implement this method.")

    def reset_state(self, batch_size: int, device: str|torch.device = None):
        pass
class ResBlock3d(BasePrimitive):
    legal_ndims = [5]
    def __init__(self, input_dim, output_dim, stride, kernel_size, act_fn, norm_num_groups, expansion, dropout):
        super(ResBlock3d, self).__init__()
        assert len(kernel_size) == 3, "kernel_size should be a tuple of 3 elements for 3D convolution"
        assert len(stride) == 3, "stride should be a tuple of 3 elements for 3D convolution"
        self.expansion = expansion
        self.conv1 = nn.Conv3d(input_dim, output_dim*self.expansion, kernel_size=kernel_size, stride=stride, padding=np.array(kernel_size)//2, bias=False)
        self.norm1 = nn.GroupNorm(norm_num_groups, output_dim*self.expansion)
        self.conv2 = nn.Conv3d(output_dim*self.expansion, output_dim, kernel_size=kernel_size, stride=1, padding=np.array(kernel_size)//2, bias=False)
        self.norm2 = nn.GroupNorm(norm_num_groups, output_dim)
        self.dropout = nn.Dropout3d(dropout)

        self.act_fn = _ACTIVATION_FUNCTIONS[act_fn]

        self.shortcut = nn.Sequential()
        if stride != 1 or input_dim != output_dim:
            self.shortcut = nn.Sequential(
                nn.Conv3d(input_dim, output_dim, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.act_fn((self.norm1(self.conv1(x))))
        out = self.norm2(self.conv2(out))
        out = self.dropout(out)
        out = out + self.shortcut(x)
        out = self.act_fn(out)
        return out
class DAGClassifier(nn.Module):
    def __init__(self, input_dim, input_len, nodes_list, parents_list, children_list, mergers_list, num_classes):
        super(DAGClassifier, self).__init__()
        self.nodes_list = nn.ModuleList(nodes_list)
        self.parents_list = parents_list
        self.children_list = children_list
        self.mergers_list = nn.ModuleList(mergers_list)

        self.leaf_nodes = [i for i, children in enumerate(children_list) if len(children) == 0]

        with torch.no_grad():
            x = torch.zeros(1, input_dim, *input_len)
            x = self.forward_dag(x)
            hidden_dim = x.shape[1]

        self.clf = nn.Sequential(*[
            nn.Linear(hidden_dim, num_classes),
        ])

    def forward_dag(self, x):
        outputs_list = list()
        for i, node in enumerate(self.nodes_list):
            parents = self.parents_list[i]
            if len(parents) == 0:
                x = node(x)
            else:
                x_list = [outputs_list[p] for p in parents]
                x = self.mergers_list[i](x_list)
                x = node(x)
            outputs_list.append(x)

        output = torch.cat([outputs_list[i].flatten(start_dim=1, end_dim=-1) for i in self.leaf_nodes], dim=1)
        return output

    def forward(self, x):
        x = self.forward_dag(x)
        x = self.clf(x)
        return x
    
    def to(self, device):
        self = super(DAGClassifier, self).to(device)
        for i, node in enumerate(self.nodes_list):
            self.nodes_list[i] = node.to(device)
        for i, merger in enumerate(self.mergers_list):
            self.mergers_list[i] = merger.to(device)
        return self
class UnListWrapper(nn.Module):
    """
    A wrapper that takes a list of tensors and returns the first tensor in the list.
    This is useful for models that expect a single tensor input but receive a list.
    """
    def __init__(self, index):
        super().__init__()
        self.index = index

    def forward(self, x_list):
        return x_list[self.index]

class ListWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x):
        out = self.module(x)
        # wrap the tensor in a one‐element list
        return [out]

PARENTS_MERGERS = {"identity": IdentityMerger}
PRIMITIVES = {"ResBlock3d": ResBlock3d, "MaxPool3d": nn.MaxPool3d, "AdaptiveMaxPool3d": nn.AdaptiveMaxPool3d}
PRIMITIVES_PARAMS = {"ResBlock3d": ["input_dim"], "MaxPool3d": [], "AdaptiveMaxPool3d": []}
def construct_dag_by_config(input_dim, input_len, config, PRIMITIVES, PRIMITIVES_PARAMS, PRIMITIVES_PARAMS_SPACE, model_key=None, time_axis=False):
    
    initial_input_dim, initial_input_len = deepcopy(input_dim), deepcopy(input_len)
    nodes_list = [nn.Identity()]
    node_names_list = ['identity']
    node_params_list = [dict()]
    parents_list = [tuple()]
    children_list = [list()]
    output_shapes_list = [(input_dim, *input_len)]
    mergers_list = [nn.Identity()]

    nodes_config = config["model"][model_key]['nodes']

    for i, node_config in enumerate(nodes_config):
        node_name = node_config['name']
        node_params = node_config['params']
        parents = tuple(node_config['parents'])
        parent_output_shapes = [output_shapes_list[p] for p in parents]
        merger = PARENTS_MERGERS[node_config['merger']](**node_config['merger_params'])
        with torch.no_grad():
            x = merger([torch.zeros(1, *output_shape) for output_shape in parent_output_shapes])
            input_dim, input_len = int(x.shape[1]), x.shape[2:]
        if 'input_dim' in PRIMITIVES_PARAMS[node_name]:
            if time_axis:
                node_params['input_dim'] = input_len[0]
            else:
                node_params['input_dim'] = input_dim
        node = PRIMITIVES[node_name](**node_params)
        with torch.no_grad():
            x = node(x)
            output_dim, output_len = x.shape[1], x.shape[2:]
        nodes_list.append(node)
        node_names_list.append(node_name)
        node_params_list.append(node_params)
        parents_list.append(parents)
        for parent in parents:
            children_list[parent].append(i)
        mergers_list.append(merger)
        children_list.append(list())
        output_shapes_list.append((output_dim, *output_len))

    return nodes_list, parents_list, children_list, mergers_list, node_names_list, node_params_list #, dag

def build_decoder(nodes, input_shape=(8,96,96), num_classes=78):
    config={"model":{"encoder":{"nodes":copy.deepcopy(nodes)}}}
    result=construct_dag_by_config(1, input_shape, config, PRIMITIVES, PRIMITIVES_PARAMS, {}, model_key="encoder")
    classifier=DAGClassifier(1, input_shape, *result[:4], num_classes)
    return ListWrapper(nn.Sequential(UnListWrapper(0), classifier))
