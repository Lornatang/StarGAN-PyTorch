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

import cv2
import torch
import torch.nn as nn

import model
from imgproc import preprocess_one_image, tensor_to_image
from utils import load_pretrained_state_dict


def main(args):
    device = torch.device(args.device)

    # Read original image
    input_tensor = preprocess_one_image(args.inputs, True, args.half, device)

    # Initialize the model
    g_model = build_model(args.model_arch_name, device)
    print(f"Build `{args.model_arch_name}` model successfully.")

    # Load model weights
    g_model = load_pretrained_state_dict(g_model, args.compile_state, args.model_weights_path)
    print(f"Load `{args.model_arch_name}` model weights `{os.path.abspath(args.model_weights_path)}` successfully.")

    # Enable half-precision inference to reduce memory usage and inference time
    if args.half:
        g_model.half()

    # Create labels
    attr2idx = {}
    for i, attr_name in enumerate(args.supported_attrs):
        attr2idx[attr_name] = i

    class_label = []
    if args.selected_attrs == "":
        raise ValueError(f"Attribute can not be empty.")

    if args.selected_attrs not in args.supported_attrs:
        raise ValueError(f"Attribute `{args.selected_attrs}` is not supported.")

    for supported_attrs_iter in range(len(args.supported_attrs)):
        index = attr2idx[args.selected_attrs]
        if supported_attrs_iter == index:
            attr_label = 1
        else:
            attr_label = 0
        class_label.append(attr_label)

    class_label = torch.FloatTensor([class_label])
    class_label = class_label.to(device, non_blocking=True)

    # Inference
    with torch.no_grad():
        output_tensor = g_model(input_tensor, class_label)

    # Save image
    output_image = tensor_to_image(output_tensor, True, args.half)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, output_image)

    print(f"StarGAN image save to `{args.output}`")


def build_model(model_arch_name: str, device: torch.device) -> nn.Module:
    # Initialize the super-resolution model
    g_model = model.__dict__[model_arch_name](in_channels=3,
                                              out_channels=3,
                                              channels=64,
                                              label_channels=5,
                                              num_rcb=6)

    g_model = g_model.to(device)

    return g_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs",
                        type=str,
                        default="./figure/Input.jpg",
                        help="Original image path.")
    parser.add_argument("--output",
                        type=str,
                        default="./figure/Input_Blond_Hair.jpg",
                        help="StarGAN image path.")
    parser.add_argument("--supported_attrs",
                        type=list,
                        default=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"],
                        help="Support all attributes name.")
    parser.add_argument("--selected_attrs",
                        type=str,
                        default="Blond_Hair",
                        help="Support all attributes name.")
    parser.add_argument("--model_arch_name",
                        type=str,
                        default="generator",
                        help="Model architecture name.")
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
