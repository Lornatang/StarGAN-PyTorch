# Copyright 2023 Lorna Author. All Rights Reserved.
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
from typing import Any

import numpy as np
import torch
from torch import nn, Tensor

__all__ = [
    "Generator", "PathDiscriminator",
    "generator", "path_discriminator",
    "GradientPenaltyLoss",
]


class _ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(_ResidualConvBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.ReLU(True),

            nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True))

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        x = self.main(x)

        x = torch.add(x, identity)

        return x


class Generator(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, channels: int = 64, label_channels: int = 5, num_rcb: int = 6):
        super(Generator, self).__init__()
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels + label_channels, channels, (7, 7), (1, 1), (3, 3), bias=False),
            nn.InstanceNorm2d(channels, affine=True, track_running_stats=True),
            nn.ReLU(True),
        )

        self.down_sampling = nn.Sequential(
            nn.Conv2d(channels, int(2 * channels), (4, 4), (2, 2), (1, 1), bias=False),
            nn.InstanceNorm2d(int(2 * channels), affine=True, track_running_stats=True),
            nn.ReLU(True),
            nn.Conv2d(int(2 * channels), int(4 * channels), (4, 4), (2, 2), (1, 1), bias=False),
            nn.InstanceNorm2d(int(4 * channels), affine=True, track_running_stats=True),
            nn.ReLU(True),
        )

        trunk = []
        for _ in range(num_rcb):
            trunk.append(_ResidualConvBlock(int(4 * channels), int(4 * channels)))
        self.trunk = nn.Sequential(*trunk)

        self.up_sampling = nn.Sequential(
            nn.ConvTranspose2d(int(4 * channels), int(2 * channels), (4, 4), (2, 2), (1, 1), bias=False),
            nn.InstanceNorm2d(int(2 * channels), affine=True, track_running_stats=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(int(2 * channels), channels, (4, 4), (2, 2), (1, 1), bias=False),
            nn.InstanceNorm2d(channels, affine=True, track_running_stats=True),
            nn.ReLU(True),
        )

        self.last_layer = nn.Sequential(
            nn.Conv2d(channels, out_channels, (7, 7), (1, 1), (3, 3), bias=False),
            nn.Tanh(),
        )

    def forward(self, x: Tensor, label: Tensor) -> Tensor:
        label = label.view(label.size(0), label.size(1), 1, 1)
        label = label.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, label], dim=1)

        x = self.first_layer(x)
        x = self.down_sampling(x)
        x = self.trunk(x)
        x = self.up_sampling(x)
        x = self.last_layer(x)

        return x


class PathDiscriminator(nn.Module):
    def __init__(
            self,
            image_size: int = 128,
            in_channels: int = 3,
            out_channels: int = 1,
            channels: int = 64,
            label_channels: int = 5,
            num_blocks: int = 6,
    ):
        super(PathDiscriminator, self).__init__()
        main = [
            nn.Conv2d(in_channels, channels, (4, 4), (2, 2), (1, 1)),
            nn.LeakyReLU(0.01, True),
        ]

        curr_channels = channels
        for _ in range(1, num_blocks):
            main.append(nn.Conv2d(curr_channels, curr_channels * 2, (4, 4), (2, 2), (1, 1)))
            main.append(nn.LeakyReLU(0.01))
            curr_channels = curr_channels * 2
        self.main = nn.Sequential(*main)

        kernel_size = int(image_size / np.power(2, num_blocks))
        self.conv1 = nn.Conv2d(curr_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)
        self.conv2 = nn.Conv2d(curr_channels, label_channels, (kernel_size, kernel_size), (1, 1), (0, 0), bias=False)

    def forward(self, x: Tensor) -> tuple[Any, Any]:
        x = self.main(x)
        x_out = self.conv1(x)
        x_class = self.conv2(x)
        x_class = x_class.view(x_class.size(0), x_class.size(1))

        return x_out, x_class


class GradientPenaltyLoss(nn.Module):
    def __init__(self):
        super(GradientPenaltyLoss, self).__init__()

    def forward(self, target: Tensor, source: Tensor) -> Tensor:
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(target.size(), device=target.device)
        dydx = torch.autograd.grad(outputs=target,
                                   inputs=source,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))

        gp_loss = torch.mean((dydx_l2norm - 1) ** 2)
        return gp_loss


def generator(**kwargs) -> Generator:
    return Generator(**kwargs)


def path_discriminator(**kwargs) -> PathDiscriminator:
    return PathDiscriminator(**kwargs)
