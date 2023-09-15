# Copyright 2023 Lorna. All Rights Reserved.
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
import shutil
from collections import OrderedDict
from enum import Enum
from typing import Any, List

import numpy as np
import torch
import torch.backends.mps
from torch import nn, Tensor
from torch.nn import Module
from torch.optim import Optimizer

__all__ = [
    "create_labels", "label2onehot", "load_state_dict", "load_pretrained_state_dict", "load_resume_state_dict", "make_directory", "save_checkpoint",
    "Summary", "AverageMeter", "ProgressMeter",
]


def create_labels(original_channels: Tensor, label_channels: int = 5, selected_attrs: list = None) -> list[Any]:
    """Generate target domain labels for debugging and testing."""
    # Get hair color indices.
    hair_color_indices = []
    for i, attr_name in enumerate(selected_attrs):
        if attr_name in ["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"]:
            hair_color_indices.append(i)

    target_color_list = []
    for i in range(label_channels):
        target_channels = original_channels.clone()
        if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
            target_channels[:, i] = 1
            for j in hair_color_indices:
                if j != i:
                    target_channels[:, j] = 0
        else:
            target_channels[:, i] = (target_channels[:, i] == 0)  # Reverse attribute value.

        target_color_list.append(target_channels.to(original_channels.device))

    return target_color_list


def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1

    return out


def load_state_dict(
        model: nn.Module,
        compile_mode: bool,
        state_dict: dict,
):
    """Load model weights and parameters

    Args:
        model (nn.Module): model
        compile_mode (bool): Enable model compilation mode, `False` means not compiled, `True` means compiled
        state_dict (dict): model weights and parameters waiting to be loaded

    Returns:
        model (nn.Module): model after loading weights and parameters
    """

    # Define compilation status keywords
    compile_state = "_orig_mod"

    # Process parameter dictionary
    model_state_dict = model.state_dict()
    new_state_dict = OrderedDict()

    # Check if the model has been compiled
    for k, v in state_dict.items():
        current_compile_state = k.split(".")[0]
        if compile_mode and current_compile_state != compile_state:
            raise RuntimeError("The model is not compiled. Please use `model = torch.compile(model)`.")

        # load the model
        if compile_mode and current_compile_state != compile_state:
            name = compile_state + "." + k
        elif not compile_mode and current_compile_state == compile_state:
            name = k[10:]
        else:
            name = k
        new_state_dict[name] = v
    state_dict = new_state_dict

    # Traverse the model parameters, load the parameters in the pre-trained model into the current model
    new_state_dict = {k: v for k, v in state_dict.items() if
                      k in model_state_dict.keys() and v.size() == model_state_dict[k].size()}

    # update model parameters
    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict)

    return model


def load_pretrained_state_dict(
        model: nn.Module,
        compile_state: bool,
        model_weights_path: str,
) -> Module:
    """Load pre-trained model weights

    Args:
        model (nn.Module): model
        compile_state (bool): model compilation state, `False` means not compiled, `True` means compiled
        model_weights_path (str): model weights path

    Returns:
        model (nn.Module): the model after loading the pre-trained model weights
    """

    checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
    state_dict = checkpoint["state_dict"]
    model = load_state_dict(model, compile_state, state_dict)
    return model


def load_resume_state_dict(
        model: nn.Module,
        ema_model: Any,
        optimizer: Optimizer,
        scheduler: Any,
        compile_state: bool,
        model_weights_path: str,
) -> Any:
    """Restore training model weights

    Args:
        model (nn.Module): model
        ema_model (nn.Module): EMA model
        optimizer (nn.optim): optimizer
        scheduler (nn.optim.lr_scheduler): learning rate scheduler
        compile_state (bool, optional): Whether the model has been compiled
        model_weights_path (str): model weights path
    Returns:
        model (nn.Module): model after loading weights
        ema_model (nn.Module): EMA model after loading weights
        start_epoch (int): start epoch
        optimizer (nn.optim): optimizer after loading weights
        scheduler (nn.optim.lr_scheduler): learning rate scheduler after loading weights
    """
    # Load model weights
    checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)

    # Load training node parameters
    start_epoch = checkpoint["epoch"]
    state_dict = checkpoint["state_dict"]
    ema_state_dict = checkpoint["ema_state_dict"] if "ema_state_dict" in checkpoint else None

    model = load_state_dict(model, compile_state, state_dict)
    if ema_state_dict is not None:
        ema_model = load_state_dict(ema_model, compile_state, ema_state_dict)

    optimizer.load_state_dict(checkpoint["optimizer"])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
        return model, ema_model, start_epoch, optimizer, scheduler
    else:
        return model, ema_model, start_epoch, optimizer


def make_directory(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def save_checkpoint(
        state_dict: dict,
        file_name: str,
        samples_dir: str,
        results_dir: str,
        last_file_name: str,
        is_last: bool = False,
) -> None:
    checkpoint_path = os.path.join(samples_dir, file_name)
    torch.save(state_dict, checkpoint_path)

    if is_last:
        shutil.copyfile(checkpoint_path, os.path.join(results_dir, last_file_name))


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
