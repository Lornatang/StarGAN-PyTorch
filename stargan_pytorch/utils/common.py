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
import random

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "create_labels", "denorm", "label2onehot", "init_seed", "select_device",
]


# Modified from `https://github.com/yunjey/stargan/blob/master/solver.py`
def create_labels(c_org: Tensor, c_dim: int = 5, dataset: str = "CelebA", selected_attrs: list = None) -> list[Tensor]:
    """Generate target domain labels for debugging and testing.

    Args:
        c_org (torch.Tensor): Original domain labels.
        c_dim (int, optional): Dimension of domain labels. Default: 5.
        dataset (str, optional): Dataset name. Default: "CelebA".
        selected_attrs (list, optional): Selected attributes. Default: None.

    Returns:
        list[torch.Tensor]: Target domain labels.
    """

    # Get hair color indices.
    global c_trg, hair_color_indices
    if dataset == "CelebA":
        hair_color_indices = []
        for i, attr_name in enumerate(selected_attrs):
            if attr_name in ["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"]:
                hair_color_indices.append(i)

    c_trg_list = []
    for i in range(c_dim):
        if dataset == "CelebA":
            c_trg = c_org.clone()
            if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                c_trg[:, i] = 1
                for j in hair_color_indices:
                    if j != i:
                        c_trg[:, j] = 0
            else:
                c_trg[:, i] = (c_trg[:, i] == 0)  # Reverse attribute value.
        elif dataset == "RaFD":
            c_trg = label2onehot(torch.ones(c_org.size(0)) * i, c_dim)

        c_trg_list.append(c_trg.to(c_org.device))
    return c_trg_list


def denorm(x: Tensor or np.ndarray) -> Tensor:
    """Convert the range from [-1, 1] to [0, 1].

    Args:
        x (torch.Tensor or np.ndarray): Input tensor or array.

    Returns:
        torch.Tensor: Output tensor.
    """

    out = (x + 1) / 2
    return out.clamp_(0, 1)


def label2onehot(labels: Tensor, dim: int) -> Tensor:
    """Convert label indices to one-hot vectors.

    Args:
        labels (torch.Tensor): Label indices.
        dim (int): Number of classes.
    """

    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out


def init_seed(seed: int = 0) -> None:
    """Initialize random seed.

    Args:
        seed (int, optional): Random seed. Default: 0.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(device: str = "") -> torch.device:
    """Select device to train model.

    Args:
        device (str, optional): Device name. Default: "".

    Returns:
        torch.device: Device to train model.
    """

    if device.lower() in ["cuda", "gpu"] and torch.cuda.is_available():
        print("Using GPU.")
        return torch.device("cuda")
    else:
        print("Using CPU.")
        return torch.device("cpu")
