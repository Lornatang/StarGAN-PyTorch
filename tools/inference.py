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
import argparse
import os

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

from stargan_pytorch.models.stargan import generator
from stargan_pytorch.models.utils import load_state_dict
from stargan_pytorch.utils.common import create_labels, denorm


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="./results/pretrained_models/g_celeba-128x128.pth.tar",
        help="Model weights file path. Default: `./results/pretrained_models/g_celeba-128x128.pth.tar`",
    )
    parser.add_argument("-i", "--inputs",
                        type=str,
                        default="./figure/example.jpg",
                        help="Input image path.")
    parser.add_argument("--img-size",
                        type=int,
                        default=128,
                        help="Image size.")
    parser.add_argument("-o", "--output",
                        type=str,
                        default="./results/example_output.jpg",
                        help="StarGAN image path.")
    parser.add_argument("--selected-attrs",
                        type=list,
                        default=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"],
                        help="Selected attributes.")
    parser.add_argument("--compile-mode",
                        action="store_true",
                        help="Whether to compile the model state.")
    parser.add_argument("--half",
                        action="store_true",
                        help="Use half precision.")
    parser.add_argument("--device",
                        type=str,
                        default="gpu",
                        choices=["cpu", "gpu"],
                        help="Device to run model. Choices: ['cpu', 'gpu']. Default: `gpu")
    opts = parser.parse_args()

    return opts


def main(opts: argparse.Namespace, device: torch.device):
    # build model
    model = generator()
    model = model.to(device)
    state_dict = torch.load(opts.weights)["state_dict"]
    model = load_state_dict(model, state_dict, opts.compile_mode)
    print(f"Load `{os.path.abspath(opts.weights)}` model weights successfully.")

    # Read original image
    transform = transforms.Compose([
        transforms.Resize(opts.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    img = Image.open(opts.inputs)
    img = transform(img).unsqueeze(0)
    img = img.to(device, non_blocking=True)

    # Enable half-precision inference to reduce memory usage and inference time
    if opts.half:
        model.half()
        img = img.half()

    # Set model to eval mode
    model.eval()

    # Disable gradient calculation
    torch.set_grad_enabled(False)

    # Create labels
    attr2idx = {}
    for i, attr_name in enumerate(opts.selected_attrs):
        attr2idx[attr_name] = i

    labels = []
    for attr_name in opts.selected_attrs:
        index = attr2idx[attr_name]
        if attr_name == index:
            attr_label = 1
        else:
            attr_label = 0
        labels.append(attr_label)

    labels = torch.FloatTensor([labels])
    labels = labels.to(device, non_blocking=True)
    labels_list = create_labels(labels, len(opts.selected_attrs), "CelebA", opts.selected_attrs)

    # Inference
    out = [img]
    for labels in labels_list:
        out.append(model(img, labels))

    # Save image
    out_concat = torch.cat(out, dim=3)
    save_image(denorm(out_concat.data.cpu()), opts.output, nrow=1, padding=0)
    print(f"Save image to `{opts.output}`")


if __name__ == "__main__":
    opts = get_opts()

    if opts.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main(opts, device)
