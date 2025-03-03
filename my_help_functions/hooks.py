# remove all
# https://discuss.pytorch.org/t/how-do-i-remove-forward-hooks-on-a-module-without-the-hook-handles/140393


from collections import OrderedDict
from typing import Dict, Callable
import torch
import torch.nn as nn

layer_outputs_fwd = {}
layer_outputs_bck = {}
bns = []

def remove_all_hooks(model: torch.nn.Module) -> None:
    global layer_outputs_fwd
    layer_outputs_fwd = {}
    global layer_outputs_bck
    layer_outputs_bck = {}
    global bns
    bns = []

    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks: Dict[int, Callable] = OrderedDict()
            elif hasattr(child, "_forward_pre_hooks"):
                child._forward_pre_hooks: Dict[int, Callable] = OrderedDict()
            elif hasattr(child, "_backward_hooks"):
                child._backward_hooks: Dict[int, Callable] = OrderedDict()
            remove_all_hooks(child)


def hook_fn_fwd(module, input, output):
    print('forward hook used')
    global layer_outputs_fwd
    layer_outputs_fwd[module] = [input[0].squeeze(0), output.squeeze(0)]

def hook_fn_bck(module, input, output):
    print('backward hook used')
    global layer_outputs_bck
    layer_outputs_bck[module] = [input, output]


def register_hooks(model, layers, backward=False):
    remove_all_hooks(model)
    for layer_to_add in layers:
        for name, layer in model.named_modules():
            if layer_to_add in name:
                print(f'add {layer_to_add}')
                layer.register_forward_hook(hook_fn_fwd)
                if backward:
                    layer.register_backward_hook(hook_fn_bck)
    if backward:
        return layer_outputs_fwd, layer_outputs_bck
    else:
        return layer_outputs_fwd


def hook_bn(module, input, output):
    global bns
    bns.append([module, input[0].squeeze(0), output.squeeze(0)])


def register_bn_hooks(model):
    remove_all_hooks(model)
    for name, layer in model.named_modules():
        if isinstance(layer, nn.BatchNorm2d):
            layer.register_forward_hook(hook_bn)
    return bns
