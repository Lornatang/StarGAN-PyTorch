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
from typing import Any

import numpy as np
from torch import nn, Tensor

__all__ = [
    "PathDiscriminator", "path_discriminator",
]


class PathDiscriminator(nn.Module):
    def __init__(
            self,
            img_size: int = 128,
            c_dim: int = 5,
            in_channels: int = 3,
            out_channels: int = 1,
            channels: int = 64,
            num_blocks: int = 6,
    ) -> None:
        """Discriminator of the PatchGAN

        Args:
            img_size (int, optional): The size of the input image. Defaults: 128.
            c_dim (int, optional): The number of channels in the label image. Defaults: 5.
            in_channels (int, optional): The number of channels in the input image. Defaults: 3.
            out_channels (int, optional): The number of channels in the output image. Defaults: 1.
            channels (int, optional): The number of channels in all conv blocks. Defaults: 64.
            num_blocks (int, optional): The number of conv blocks in the discriminator. Defaults: 6.

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

        kernel_size = int(img_size / np.power(2, num_blocks))
        self.conv1 = nn.Conv2d(curr_channels, out_channels, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(curr_channels, c_dim, (kernel_size, kernel_size), 1, (0, 0), bias=False)

    def forward(self, x: Tensor) -> tuple[Any, Any]:
        x = self.main(x)
        x_out = self.conv1(x)
        x_cls = self.conv2(x)
        x_cls = x_cls.view(x_cls.size(0), x_cls.size(1))

        return x_out, x_cls


def path_discriminator(img_size: int = 128, c_dim: int = 5, **kwargs) -> PathDiscriminator:
    return PathDiscriminator(img_size=img_size, c_dim=c_dim, **kwargs)
