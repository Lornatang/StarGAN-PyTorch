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
from torch.utils.data import Dataset
from torchvision import transforms

__all__ = [
    "CelebADataset"
]


class CelebADataset(Dataset):
    def __init__(
            self,
            images_dir: str,
            transforms: transforms.Compose,
            attr_path: str,
            selected_attrs: str,
    ) -> None:
        self.images_dir = images_dir
        self.transforms = transforms
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs

        self.image_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}

        # Preprocess attribute file
        lines = [line.rstrip() for line in open(self.attr_path, "r")]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == "1")

            self.image_dataset.append([filename, label])

    def __getitem__(self, batch_index: int) -> (torch.Tensor, torch.Tensor):
        """Return one image and its corresponding attribute label."""
        filename, label = self.image_dataset[batch_index]

        image = Image.open(os.path.join(self.images_dir, filename))

        image_tensor = self.transforms(image)
        label_tensor = torch.FloatTensor(label)

        return {"image": image_tensor, "label": label_tensor}

    def __len__(self):
        """Return the number of images."""
        return len(self.image_dataset)
