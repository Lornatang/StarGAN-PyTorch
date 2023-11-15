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
        """Residual Convolution Block

        Args:
            in_channels (int): The number of channels in the input image.
            out_channels (int): The number of channels in the output image.

        """
        super(_ResidualConvBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True),
            nn.ReLU(True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True, track_running_stats=True))

    def forward(self, x: Tensor) -> Tensor:
        return self.main(x) + x


class Generator(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
            num_rcb: int = 6,
            c_dim: int = 5,
    ) -> None:
        """Generator of the Pix2Pix

        Args:
            in_channels (int, optional): The number of channels in the input image. Defaults: 3.
            out_channels (int, optional): The number of channels in the output image. Defaults: 3.
            channels (int, optional): The number of channels in all conv blocks. Defaults: 64.
            num_rcb (int, optional): The number of residual conv blocks in the generator. Defaults: 6.
            c_dim (int, optional): The number of channels in the label image. Defaults: 5.

        """
        super(Generator, self).__init__()
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels + c_dim, channels, 7, 1, 3, bias=False),
            nn.InstanceNorm2d(channels, affine=True, track_running_stats=True),
            nn.ReLU(True),
        )

        self.down_sampling = nn.Sequential(
            nn.Conv2d(channels, int(2 * channels), 4, 2, 1, bias=False),
            nn.InstanceNorm2d(int(2 * channels), affine=True, track_running_stats=True),
            nn.ReLU(True),
            nn.Conv2d(int(2 * channels), int(4 * channels), 4, 2, 1, bias=False),
            nn.InstanceNorm2d(int(4 * channels), affine=True, track_running_stats=True),
            nn.ReLU(True),
        )

        trunk = []
        for _ in range(num_rcb):
            trunk.append(_ResidualConvBlock(int(4 * channels), int(4 * channels)))
        self.trunk = nn.Sequential(*trunk)

        self.up_sampling = nn.Sequential(
            nn.ConvTranspose2d(int(4 * channels), int(2 * channels), 4, 2, 1, bias=False),
            nn.InstanceNorm2d(int(2 * channels), affine=True, track_running_stats=True),
            nn.ReLU(True),
            nn.ConvTranspose2d(int(2 * channels), channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(channels, affine=True, track_running_stats=True),
            nn.ReLU(True),
        )

        self.last_layer = nn.Sequential(
            nn.Conv2d(channels, out_channels, 7, 1, 3, bias=False),
            nn.Tanh(),
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)

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
            num_blocks: int = 6,
            c_dim: int = 5,
    ) -> None:
        """Discriminator of the PatchGAN

        Args:
            image_size (int, optional): The size of the input image. Defaults: 128.
            in_channels (int, optional): The number of channels in the input image. Defaults: 3.
            out_channels (int, optional): The number of channels in the output image. Defaults: 1.
            channels (int, optional): The number of channels in all conv blocks. Defaults: 64.
            num_blocks (int, optional): The number of conv blocks in the discriminator. Defaults: 6.
            c_dim (int, optional): The number of channels in the label image. Defaults: 5.

        """
        super(PathDiscriminator, self).__init__()
        main = [
            nn.Conv2d(in_channels, channels, 4, 2, 1),
            nn.LeakyReLU(0.01, True),
        ]

        curr_channels = channels
        for _ in range(1, num_blocks):
            main.append(nn.Conv2d(curr_channels, curr_channels * 2, 4, 2, 1))
            main.append(nn.LeakyReLU(0.01))
            curr_channels = curr_channels * 2
        self.main = nn.Sequential(*main)

        kernel_size = int(image_size / np.power(2, num_blocks))
        self.conv1 = nn.Conv2d(curr_channels, out_channels, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(curr_channels, c_dim, (kernel_size, kernel_size), 1, (0, 0), bias=False)

    def forward(self, x: Tensor) -> tuple[Any, Any]:
        x = self.main(x)
        x_out = self.conv1(x)
        x_cls = self.conv2(x)
        x_cls = x_cls.view(x_cls.size(0), x_cls.size(1))

        return x_out, x_cls


class GradientPenaltyLoss(nn.Module):
    def __init__(self):
        super(GradientPenaltyLoss, self).__init__()

    def forward(self, target: Tensor, source: Tensor) -> Tensor:
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2.

        Args:
            target (Tensor): The output of the discriminator with respect to the real image.
            source (Tensor): The output of the discriminator with respect to the fake image.

        """
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
