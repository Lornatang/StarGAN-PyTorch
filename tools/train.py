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
"""
Training master program.
All training scripts are scheduled by this script
"""
import argparse
import os
import random

import numpy as np
import torch
import yaml
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

from stargan_pytorch.engine.trainer import Trainer


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch StarGAN Training")
    parser.add_argument("config_path", metavar="FILE", help="path to config file")
    opts = parser.parse_args()

    return opts


def init(config) -> tuple:
    # Fixed random number seed
    seed = config["SEED"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Because the size of the input image is fixed, the fixed CUDNN convolution method can greatly increase the running speed
    cudnn.benchmark = True

    # Define the running device number
    device = torch.device("cuda", config["DEVICE_ID"])

    # Create a folder to save the model and log
    save_weights_dir = os.path.join("results", "train", config["EXP_NAME"])
    save_visuals_dir = os.path.join("results", "train", config["EXP_NAME"], "visuals")
    save_tblogger_dir = os.path.join("tb_logger", config["EXP_NAME"])
    os.makedirs(save_weights_dir, exist_ok=True)
    os.makedirs(save_visuals_dir, exist_ok=True)
    os.makedirs(save_tblogger_dir, exist_ok=True)

    # Use tensorboard to record the training process
    tblogger = SummaryWriter(save_tblogger_dir)

    return device, save_weights_dir, save_visuals_dir, tblogger


def main() -> None:
    opts = get_opts()

    # Read YAML configuration file
    with open(opts.config_path, "r") as f:
        config = yaml.full_load(f)

    device, save_weights_dir, save_visuals_dir, tblogger = init(config)

    app = Trainer(config, device, save_weights_dir, save_visuals_dir, tblogger)
    app.train()


if __name__ == "__main__":
    main()
