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

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

__all__ = [
    "CelebADataset"
]


class CelebADataset(Dataset):
    def __init__(
            self,
            root: str,
            transforms: transforms.Compose,
            attr_path: str,
            selected_attrs: str,
    ) -> None:
        self.root = root
        self.transforms = transforms
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs

        self.img_dataset = []
        self.attr2idx = {}

        # Preprocess attribute file
        lines = [line.rstrip() for line in open(self.attr_path, "r")]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i

        lines = lines[2:]
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == "1")

            self.img_dataset.append([filename, label])

    def __getitem__(self, batch_index: int) -> (Tensor, Tensor):
        """Return one image and its corresponding attribute label."""
        file_name, label = self.img_dataset[batch_index]

        img = Image.open(os.path.join(self.root, file_name))

        img = self.transforms(img)
        label = torch.FloatTensor(label)

        return img, label

    def __len__(self):
        """Return the number of images."""
        return len(self.img_dataset)
