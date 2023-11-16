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
import torch
from torch import nn, Tensor

__all__ = [
    "StarGAN", "generator",
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


class StarGAN(nn.Module):
    def __init__(
            self,
            c_dim: int = 5,
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
            num_rcb: int = 6,
    ) -> None:
        """Generator of the Pix2Pix

        Args:
            c_dim (int, optional): The number of channels in the label image. Defaults: 5.
            in_channels (int, optional): The number of channels in the input image. Defaults: 3.
            out_channels (int, optional): The number of channels in the output image. Defaults: 3.
            channels (int, optional): The number of channels in all conv blocks. Defaults: 64.
            num_rcb (int, optional): The number of residual conv blocks in the generator. Defaults: 6.

        """
        super(StarGAN, self).__init__()
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


def generator(c_dim: int = 5, **kwargs) -> StarGAN:
    return StarGAN(c_dim=c_dim, **kwargs)
