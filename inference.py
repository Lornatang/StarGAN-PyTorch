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
import argparse
import os

import torch
import torch.nn as nn
from torchvision.utils import save_image

from model import generator
from imgproc import preprocess_one_image
from utils import create_labels, denorm, load_pretrained_state_dict


def main(args):
    device = torch.device(args.device)

    # Read original image
    input_tensor = preprocess_one_image(args.inputs, args.image_size, device)

    # Initialize the model
    model = build_model(device)

    # Load model weights
    model = load_pretrained_state_dict(model, args.compile_state, args.model_weights_path)
    print(f"Load `{os.path.abspath(args.model_weights_path)}` model weights successfully.")

    # Enable half-precision inference to reduce memory usage and inference time
    if args.half:
        model.half()

    # Create labels
    attr2idx = {}
    for i, attr_name in enumerate(args.supported_attrs_name):
        attr2idx[attr_name] = i

    original_channels = []
    for attr_name in args.supported_attrs_name:
        index = attr2idx[attr_name]
        if attr_name == index:
            attr_label = 1
        else:
            attr_label = 0
        original_channels.append(attr_label)

    original_channels = torch.FloatTensor([original_channels])
    original_channels = original_channels.to(device, non_blocking=True)
    labels = create_labels(original_channels, len(args.supported_attrs_name), args.dataset, args.supported_attrs_name)

    # Inference
    with torch.no_grad():
        output_tensor = [input_tensor]
        for label in labels:
            output_tensor.append(model(input_tensor, label))

    # Save image
    x_concat = torch.cat(output_tensor, dim=3)
    save_image(denorm(x_concat.data.cpu()), args.output, nrows=1, padding=0)
    print(f"StarGAN image save to `{args.output}`")


def build_model(device: torch.device) -> nn.Module:
    model = generator()
    model = model.to(device)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputs",
                        type=str,
                        default="./figure/celeba_example1.jpg",
                        help="Original image path.")
    parser.add_argument("--image_size",
                        type=int,
                        default=128,
                        help="Image size.")
    parser.add_argument("-o", "--output",
                        type=str,
                        default="./figure/results_celeba_example1.jpg",
                        help="StarGAN image path.")
    parser.add_argument("--dataset",
                        type=str,
                        default="CelebA",
                        help="Dataset name.")
    parser.add_argument("--supported_attrs_name",
                        type=list,
                        default=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"],
                        help="Support all attributes name.")
    parser.add_argument("--compile_state",
                        type=bool,
                        default=False,
                        help="Whether to compile the model state.")
    parser.add_argument("--model_weights_path",
                        type=str,
                        default="./results/pretrained_models/StarGAN-CelebA-128x128.pth.tar",
                        help="Model weights file path.")
    parser.add_argument("--half",
                        action="store_true",
                        help="Use half precision.")
    parser.add_argument("--device",
                        type=str,
                        default="cuda:0",
                        help="Device to run model.")
    args = parser.parse_args()

    main(args)
