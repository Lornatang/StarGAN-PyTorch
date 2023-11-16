# Copyright 2023 AlphaBetter Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import warnings
from collections import OrderedDict
from pathlib import Path

import thop
import torch
from torch import nn, optim, Tensor

__all__ = [
    "load_state_dict", "load_resume_state_dict", "profile"
]


def load_state_dict(
        model: nn.Module,
        state_dict: dict,
        compile_mode: bool = False,
) -> nn.Module:
    """Load the PyTorch model weights from the model weight address

    Args:
        model (nn.Module): PyTorch model
        state_dict: PyTorch model state dict
        compile_mode (bool, optional): Compile mode. Default: ``False``

    Returns:
        model: PyTorch model with weights
    """

    # When the PyTorch version is less than 2.0, the model compilation is not supported.
    if int(torch.__version__[0]) < 2 and compile_mode:
        warnings.warn("PyTorch version is less than 2.0, does not support model compilation.")
        compile_mode = False

    # compile keyword
    compile_keyword = "_orig_mod"

    # Create new OrderedDict that does not contain the module prefix
    model_state_dict = model.state_dict()
    new_state_dict = OrderedDict()

    # Remove the module prefix and update the model weight
    for k, v in state_dict.items():
        k_prefix = k.split(".")[0]

        if k_prefix == compile_keyword and not compile_mode:
            name = k[len(compile_keyword) + 1:]
        elif k_prefix != compile_keyword and compile_mode:
            raise ValueError("The model is not compiled, but the weight is compiled.")
        else:
            name = k
        new_state_dict[name] = v
    state_dict = new_state_dict

    # Filter out unnecessary parameters
    new_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}

    # Update model parameters
    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict)

    return model


def load_resume_state_dict(
        model: nn.Module,
        ema_model: nn.Module = None,
        model_weights_path: str | Path = "",
        compile_mode: bool = False,
) -> tuple[int, nn.Module, nn.Module, optim.Optimizer, optim.lr_scheduler]:
    """Load the PyTorch model weights from the model weight address

    Args:
        model (nn.Module): PyTorch model
        model_weights_path: PyTorch model path
        ema_model (nn.Module): EMA model
        compile_mode (bool, optional): Compile mode. Default: ``False``

    Returns:
        start_epoch (int): Start epoch
        best_mean_ap (float): Best mean ap
        model: PyTorch model with weights
        model_weights_path: PyTorch model path
        ema_model (nn.Module): EMA model
        optimizer (optim.Optimizer): Optimizer
        scheduler (optim.lr_scheduler): Scheduler
    """

    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"Model weights file not found '{model_weights_path}'")

    checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
    start_epoch = checkpoint["epoch"]
    model = load_state_dict(model, checkpoint["state_dict"], compile_mode)
    ema_model = load_state_dict(ema_model, checkpoint["ema_state_dict"], compile_mode) if ema_model is not None else None
    optimizer = checkpoint["optimizer"]
    scheduler = checkpoint["scheduler"]

    return start_epoch, model, ema_model, optimizer, scheduler


def profile(model: nn.Module, inputs: Tensor, device: str | torch.device = "cpu", verbose: bool = False) -> tuple[float, float, float]:
    """Profile model

    Args:
        model (nn.Module): PyTorch model
        inputs (Tensor): Inputs
        device (str or torch.device, optional): Device. Default: ``cpu``
        verbose (bool, optional): Verbose. Default: ``False``

    Returns:
        flops (float): FLOPs
        memory (float): Memory
        params (float): Params
    """

    if not isinstance(device, torch.device):
        device = torch.device(device)

    model.eval()
    model.to(device)

    inputs = torch.Tensor(inputs).to(device)

    flops = thop.profile(model, inputs=(inputs,), verbose=False)[0] / 1E9 * 2  # GFLOPs
    memory = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)
    parameters = sum(x.numel() for x in model.parameters()) if isinstance(model, nn.Module) / 1E6 else 0

    torch.cuda.empty_cache()

    if verbose:
        print(f"FLOPs: {flops:.3f} GFLOPs\n"
              f"Memory: {memory:.3f} GB\n"
              f"Params: {parameters:.3f} M")

    return flops, memory, parameters
