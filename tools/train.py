# Copyright 2023 Lornatang Authors. All Rights Reserved.
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
import time

import torch
import wandb
import yaml
from torch.backends import cudnn
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter

from stargan_pytorch.engine.trainer import Trainer
from stargan_pytorch.utils import init_seed, select_device


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch StarGAN Training")
    parser.add_argument("config_path", metavar="FILE", help="path to config file")
    opts = parser.parse_args()

    return opts


def init(config) -> tuple:
    # Fixed random number seed
    init_seed(config["SEED"])

    # Define the running device number
    device = select_device(config["DEVICE"])

    if device.type != "cuda":
        raise RuntimeError("Must use CUDA for training")

    # Because the size of the input image is fixed, the fixed CUDNN convolution method can greatly increase the running speed
    cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    # Initialize the mixed precision method
    scaler = amp.GradScaler()

    # Create a folder to save the model and log
    strtime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    save_weights_dir = os.path.join("results", "train", config["EXP_NAME"] + "-" + strtime)
    save_visuals_dir = os.path.join("results", "train", config["EXP_NAME"] + "-" + strtime, "visuals")
    save_tblogger_dir = os.path.join("tb_logger", config["EXP_NAME"] + "-" + strtime)
    os.makedirs(save_weights_dir, exist_ok=True)
    os.makedirs(save_visuals_dir, exist_ok=True)
    os.makedirs(save_tblogger_dir, exist_ok=True)

    # Use tensorboard to record the training process
    tblogger = SummaryWriter(save_tblogger_dir)

    # Use wandb to record the training process
    wandb_project_name = config["PROJECT_NAME"]
    wandb_name = config["EXP_NAME"] + "-" + strtime
    wandb.init(config=config, project=wandb_project_name, name=wandb_name)

    return scaler, device, save_weights_dir, save_visuals_dir, tblogger


def main() -> None:
    opts = get_opts()

    # Read YAML configuration file
    with open(opts.config_path, "r") as f:
        config = yaml.full_load(f)

    scaler, device, save_weights_dir, save_visuals_dir, tblogger = init(config)

    app = Trainer(config, scaler, device, save_weights_dir, save_visuals_dir, tblogger)
    app.train()


if __name__ == "__main__":
    main()
