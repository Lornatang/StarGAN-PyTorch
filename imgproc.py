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
import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms

__all__ = [
    "preprocess_one_image",
]


def preprocess_one_image(image_path: str, image_size: int, device: torch.device) -> Tensor:
    image = Image.open(image_path)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Un-Squeeze to add a batch dimension
    tensor = transform(image).unsqueeze(0)

    # Data transfer to the specified device
    tensor = tensor.to(device, non_blocking=True)

    return tensor
