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
import random
import time
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.backends import cudnn
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import model
from dataset import CUDAPrefetcher, CelebADataset
from utils import load_resume_state_dict, load_pretrained_state_dict, make_directory, save_checkpoint, Summary, AverageMeter, ProgressMeter


def main():
    # Read parameters from configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",
                        type=str,
                        default="./configs/train/CelebA.yaml",
                        help="Path to train config file.")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.full_load(f)

    # Fixed random number seed
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    torch.manual_seed(config["SEED"])
    torch.cuda.manual_seed_all(config["SEED"])

    # Because the size of the input image is fixed, the fixed CUDNN convolution method can greatly increase the running speed
    cudnn.benchmark = True

    # Initialize the mixed precision method
    scaler = amp.GradScaler()

    # Default to start training from scratch
    start_epoch = 0

    # Define the running device number
    device = torch.device("cuda", config["DEVICE_ID"])

    # Define the basic functions needed to start training
    train_data_prefetcher = load_dataset(config, device)
    g_model, ema_g_model, d_model = build_model(config, device)
    classification_criterion, gp_criterion, rec_criterion = define_loss(config, device)
    g_optimizer, d_optimizer = define_optimizer(g_model, d_model, config)
    g_scheduler, d_scheduler = define_scheduler(g_optimizer, d_optimizer, config)

    # Load the pretrained model
    if config["TRAIN"]["CHECKPOINT"]["PRETRAINED_G_MODEL"]:
        g_model = load_pretrained_state_dict(g_model,
                                             config["MODEL"]["G"]["COMPILED"],
                                             config["TRAIN"]["CHECKPOINT"]["PRETRAINED_G_MODEL"])
        print(f"Loaded `{config['TRAIN']['CHECKPOINT']['PRETRAINED_G_MODEL']}` pretrained model weights successfully.")
    else:
        print("Pretrained g model weights not found.")
    if config["TRAIN"]["CHECKPOINT"]["PRETRAINED_D_MODEL"]:
        d_model = load_pretrained_state_dict(d_model,
                                             config["MODEL"]["D"]["COMPILED"],
                                             config["TRAIN"]["CHECKPOINT"]["PRETRAINED_D_MODEL"])
        print(f"Loaded `{config['TRAIN']['CHECKPOINT']['PRETRAINED_D_MODEL']}` pretrained model weights successfully.")
    else:
        print("Pretrained dd model weights not found.")

    # Load the last training interruption model node
    if config["TRAIN"]["CHECKPOINT"]["RESUMED_G_MODEL"]:
        g_model, ema_g_model, start_epoch, best_psnr, best_ssim, g_optimizer, g_scheduler = load_resume_state_dict(
            g_model,
            ema_g_model,
            g_optimizer,
            g_scheduler,
            config["MODEL"]["G"]["COMPILED"],
            config["TRAIN"]["CHECKPOINT"]["RESUMED_G_MODEL"],
        )
        print(f"Loaded `{config['TRAIN']['CHECKPOINT']['RESUMED_G_MODEL']}` resume model weights successfully.")
    else:
        print("Resume training g model not found. Start training from scratch.")
    if config["TRAIN"]["CHECKPOINT"]["RESUMED_D_MODEL"]:
        d_model, _, start_epoch, best_psnr, best_ssim, d_optimizer, d_scheduler = load_resume_state_dict(
            d_model,
            None,
            d_optimizer,
            d_scheduler,
            config["MODEL"]["D"]["COMPILED"],
            config["TRAIN"]["CHECKPOINT"]["RESUMED_D_MODEL"],
        )
        print(f"Loaded `{config['TRAIN']['CHECKPOINT']['RESUMED_D_MODEL']}` resume model weights successfully.")
    else:
        print("Resume training d model not found. Start training from scratch.")

    # Create the folder where the model weights are saved
    samples_dir = os.path.join("samples", config["EXP_NAME"])
    results_dir = os.path.join("results", config["EXP_NAME"])
    make_directory(samples_dir)
    make_directory(results_dir)

    # create model training log
    writer = SummaryWriter(os.path.join("samples", "logs", config["EXP_NAME"]))

    for epoch in range(start_epoch, config["TRAIN"]["HYP"]["EPOCHS"]):
        train(g_model,
              ema_g_model,
              d_model,
              train_data_prefetcher,
              classification_criterion,
              gp_criterion,
              rec_criterion,
              g_optimizer,
              g_optimizer,
              scaler,
              writer,
              device,
              config)

        # Update LR
        g_scheduler.step()
        d_scheduler.step()

        print("\n")

        # Automatically save model weights
        is_last = (epoch + 1) == config["TRAIN"]["HYP"]["EPOCHS"]
        save_checkpoint({"epoch": epoch + 1,
                         "state_dict": g_model.state_dict(),
                         "ema_state_dict": ema_g_model.state_dict() if ema_g_model is not None else None,
                         "optimizer": g_optimizer.state_dict()},
                        f"epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "g_last.pth.tar",
                        is_last)
        save_checkpoint({"epoch": epoch + 1,
                         "state_dict": d_model.state_dict(),
                         "optimizer": d_optimizer.state_dict()},
                        f"epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "d_last.pth.tar",
                        is_last)


def load_dataset(
        config: Any,
        device: torch.device,
) -> CUDAPrefetcher:
    # Load the train dataset
    train_datasets = CelebADataset(
        config["TRAIN"]["DATASET"]["IMAGES_DIR"],
        config["TRAIN"]["DATASET"]["CROP_IMAGE_SIZE"],
        config["TRAIN"]["DATASET"]["RESIZE_IMAGE_SIZE"],
        config["TRAIN"]["DATASET"]["ATTR_PATH"],
        config["TRAIN"]["DATASET"]["SELECTED_ATTRS"],
    )
    # generate dataset iterator
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=config["TRAIN"]["HYP"]["IMGS_PER_BATCH"],
                                  shuffle=config["TRAIN"]["HYP"]["SHUFFLE"],
                                  num_workers=config["TRAIN"]["HYP"]["NUM_WORKERS"],
                                  pin_memory=config["TRAIN"]["HYP"]["PIN_MEMORY"],
                                  drop_last=True,
                                  persistent_workers=config["TRAIN"]["HYP"]["PERSISTENT_WORKERS"])
    # Replace the data set iterator with CUDA to speed up
    train_data_prefetcher = CUDAPrefetcher(train_dataloader, device)

    return train_data_prefetcher


def build_model(
        config: Any,
        device: torch.device,
) -> [nn.Module, nn.Module or Any, nn.Module]:
    g_model = model.__dict__[config["MODEL"]["G"]["NAME"]](in_channels=config["MODEL"]["G"]["IN_CHANNELS"],
                                                           out_channels=config["MODEL"]["G"]["OUT_CHANNELS"],
                                                           channels=config["MODEL"]["G"]["CHANNELS"],
                                                           label_channels=config["MODEL"]["G"]["LABEL_CHANNELS"],
                                                           num_rcb=config["MODEL"]["G"]["NUM_RCB"])
    d_model = model.__dict__[config["MODEL"]["D"]["NAME"]](image_size=config["MODEL"]["D"]["IMAGE_SIZE"],
                                                           in_channels=config["MODEL"]["D"]["IN_CHANNELS"],
                                                           out_channels=config["MODEL"]["D"]["OUT_CHANNELS"],
                                                           channels=config["MODEL"]["D"]["CHANNELS"],
                                                           label_channels=config["MODEL"]["D"]["LABEL_CHANNELS"],
                                                           num_blocks=config["MODEL"]["D"]["NUM_BLOCKS"])

    g_model = g_model.to(device)
    d_model = d_model.to(device)

    if config["MODEL"]["EMA"]["ENABLE"]:
        # Generate an exponential average model based on a generator to stabilize model training
        ema_decay = config["MODEL"]["EMA"]["DECAY"]
        ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: \
            (1 - ema_decay) * averaged_model_parameter + ema_decay * model_parameter
        ema_g_model = AveragedModel(g_model, device=device, avg_fn=ema_avg_fn)
    else:
        ema_g_model = None

    # compile model
    if config["MODEL"]["G"]["COMPILED"]:
        g_model = torch.compile(g_model)
    if config["MODEL"]["D"]["COMPILED"]:
        d_model = torch.compile(d_model)
    if config["MODEL"]["EMA"]["COMPILED"] and ema_g_model is not None:
        ema_g_model = torch.compile(ema_g_model)

    return g_model, ema_g_model, d_model


def define_loss(config: Any, device: torch.device) -> [nn.BCEWithLogitsLoss, model.GradientPenaltyLoss, nn.L1Loss or nn.MSELoss]:
    if config["TRAIN"]["LOSSES"]["CLASSIFICATION_LOSS"]["NAME"] == "binary_cross_entropy_with_logits":
        classification_criterion = nn.BCEWithLogitsLoss()
    else:
        raise NotImplementedError(f"Loss {config['TRAIN']['LOSSES']['CLASSIFICATION_LOSS']['NAME']} is not implemented.")

    if config["TRAIN"]["LOSSES"]["GRADIENT_LOSS"]["NAME"] == "gradient_penalty":
        gp_criterion = model.GradientPenaltyLoss()
    else:
        raise NotImplementedError(f"Loss {config['TRAIN']['LOSSES']['GRADIENT_LOSS']['NAME']} is not implemented.")

    if config["TRAIN"]["LOSSES"]["REC_LOSS"]["NAME"] == "l1":
        rec_criterion = nn.L1Loss()
    elif config["TRAIN"]["LOSSES"]["REC_LOSS"]["NAME"] == "l2":
        rec_criterion = nn.MSELoss()
    else:
        raise NotImplementedError(f"Loss {config['TRAIN']['LOSSES']['REC_LOSS']['NAME']} is not implemented.")

    classification_criterion = classification_criterion.to(device)
    gp_criterion = gp_criterion.to(device)
    rec_criterion = rec_criterion.to(device)

    return classification_criterion, gp_criterion, rec_criterion


def define_optimizer(g_model: nn.Module, d_model: nn.Module, config: Any) -> [optim.Adam, optim.Adam]:
    if config["TRAIN"]["OPTIM"]["NAME"] == "Adam":
        g_optimizer = optim.Adam(g_model.parameters(),
                                 config["TRAIN"]["OPTIM"]["LR"],
                                 config["TRAIN"]["OPTIM"]["BETAS"],
                                 config["TRAIN"]["OPTIM"]["EPS"],
                                 config["TRAIN"]["OPTIM"]["WEIGHT_DECAY"])
        d_optimizer = optim.Adam(d_model.parameters(),
                                 config["TRAIN"]["OPTIM"]["LR"],
                                 config["TRAIN"]["OPTIM"]["BETAS"],
                                 config["TRAIN"]["OPTIM"]["EPS"],
                                 config["TRAIN"]["OPTIM"]["WEIGHT_DECAY"])

    else:
        raise NotImplementedError(f"Optimizer {config['TRAIN']['OPTIM']['NAME']} is not implemented.")

    return g_optimizer, d_optimizer


def define_scheduler(g_optimizer: optim.Adam, d_optimizer: optim.Adam, config: Any) -> [lr_scheduler.MultiStepLR, lr_scheduler.MultiStepLR]:
    if config["TRAIN"]["LR_SCHEDULER"]["NAME"] == "StepLR":
        g_scheduler = lr_scheduler.StepLR(g_optimizer,
                                          config["TRAIN"]["LR_SCHEDULER"]["STEP_SIZE"],
                                          config["TRAIN"]["LR_SCHEDULER"]["GAMMA"])
        d_scheduler = lr_scheduler.StepLR(d_optimizer,
                                          config["TRAIN"]["LR_SCHEDULER"]["STEP_SIZE"],
                                          config["TRAIN"]["LR_SCHEDULER"]["GAMMA"])

    else:
        raise NotImplementedError(f"LR Scheduler {config['TRAIN']['LR_SCHEDULER']['NAME']} is not implemented.")

    return g_scheduler, d_scheduler


def train(
        g_model: nn.Module,
        ema_g_model: nn.Module,
        d_model: nn.Module,
        train_data_prefetcher: CUDAPrefetcher,
        classification_criterion: nn.BCEWithLogitsLoss,
        gp_criterion: model.GradientPenaltyLoss,
        rec_criterion: nn.L1Loss or nn.MSELoss,
        g_optimizer: optim.Adam,
        d_optimizer: optim.Adam,
        scaler: amp.GradScaler,
        writer: SummaryWriter,
        device: torch.device,
        config: Any,
) -> None:
    # Calculate how many batches of data there are under a dataset iterator
    batches = len(train_data_prefetcher)

    # The information printed by the progress bar
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    data_time = AverageMeter("Data", ":6.3f", Summary.NONE)
    g_losses = AverageMeter("G Loss", ":6.6f", Summary.NONE)
    d_losses = AverageMeter("D Loss", ":6.6f", Summary.NONE)
    progress = ProgressMeter(batches,
                             [batch_time, data_time, g_losses, d_losses],
                             prefix=f"Epoch: [{config['TRAIN']['HYP']['EPOCHS'] + 1}]")

    # Set the model to training mode
    g_model.train()
    d_model.train()

    # Define loss function weights
    classification_weight = torch.Tensor(config["TRAIN"]["LOSSES"]["CLASSIFICATION_LOSS"]["WEIGHT"]).to(device)
    gp_weight = torch.Tensor(config["TRAIN"]["LOSSES"]["GRADIENT_LOSS"]["WEIGHT"]).to(device)
    rec_weight = torch.Tensor(config["TRAIN"]["LOSSES"]["REC_LOSS"]["WEIGHT"]).to(device)

    # Initialize data batches
    batch_index = 0
    # Set the dataset iterator pointer to 0
    train_data_prefetcher.reset()
    # Record the start time of training a batch
    end = time.time()
    # load the first batch of data
    batch_data = train_data_prefetcher.next()

    while batch_data is not None:
        # Load batches of data
        real_images = batch_data["image"].to(device, non_blocking=True)
        source_class_label = batch_data["label"].to(device, non_blocking=True)

        # Generate target domain labels randomly.
        rand_index = torch.randperm(source_class_label.size(0))
        target_class_label = source_class_label[rand_index]

        source_domain_label = source_class_label.clone()
        target_domain_label = target_class_label.clone()

        # Record the time to load a batch of data
        data_time.update(time.time() - end)

        # Start training the discriminator model
        # During discriminator model training, enable discriminator model backpropagation
        for d_parameters in d_model.parameters():
            d_parameters.requires_grad = True

        # Initialize the discriminator model gradient
        d_model.zero_grad(set_to_none=True)

        # Calculate the classification score of the discriminator model on real samples
        with amp.autocast():
            real_output, real_class = d_model(real_images)
            d_loss_real = - torch.mean(real_output)
            d_loss_class = classification_criterion(real_class, source_class_label)
            d_loss_class = torch.sum(torch.mul(classification_weight, d_loss_class))

        # Calculate the classification score of the generated samples by the discriminator model
        with amp.autocast():
            fake_images = g_model(real_images, target_domain_label)
            fake_output, fake_class = d_model(fake_images.detach().clone())
            d_loss_fake = torch.mean(fake_output)
            print("OK1")

            # Compute loss for gradient penalty.
            alpha = torch.rand(real_images.size(0), 1, 1, 1, device=device)
            x_hat = (alpha * real_images.data + (1 - alpha) * fake_images.data).requires_grad_(True)
            output, _ = d_model(x_hat)
            d_loss_gp = gp_criterion(output, x_hat)
            d_loss_gp = torch.sum(torch.mul(gp_weight, d_loss_gp))
            print("OK2")

        # Compute the discriminator total loss value
        print(d_loss_real)
        print(d_loss_fake)
        print(d_loss_class)
        print(d_loss_gp)
        d_loss = d_loss_real + d_loss_fake + d_loss_class + d_loss_gp
        print(d_loss)
        # backpropagate discriminator loss on generated samples
        scaler.scale(d_loss).backward()
        # Update discriminator model weights
        scaler.step(d_optimizer)
        scaler.update()
        # end training discriminator model
        print("OK3")

        # start training the generator model
        if (batch_index + 1) % config["TRAIN"]["N_CRITIC"] == 0:
            # Disable discriminator backpropagation during generator training
            for d_parameters in d_model.parameters():
                d_parameters.requires_grad = False

            # Initialize the generator model gradient
            g_model.zero_grad(set_to_none=True)

            # Calculate the perceptual loss of the generator, mainly including pixel loss, feature loss and confrontation loss
            with amp.autocast():
                fake_images = g_model(real_images, target_domain_label)
                fake_output, fake_class = d_model(fake_images)
                g_loss_fake = - torch.mean(fake_output)
                g_loss_class = classification_criterion(fake_class, target_class_label)
                g_loss_class = torch.sum(torch.mul(classification_weight, g_loss_class))

                # Target-to-original domain.
                rec_images = g_model(fake_images, source_domain_label)
                g_loss_rec = rec_criterion(rec_images, real_images)
                g_loss_rec = torch.sum(torch.mul(rec_weight, g_loss_rec))

            # Compute the generator total loss value
            g_loss = g_loss_fake + g_loss_rec + g_loss_class
            # Backpropagation generator loss on generated samples
            scaler.scale(g_loss).backward()
            # update generator model weights
            scaler.step(g_optimizer)
            scaler.update()
            # end training generator model

            if config["MODEL"]["EMA"]["ENABLE"]:
                # update exponentially averaged model weights
                ema_g_model.update_parameters(g_model)

            # record the loss value
            batch_size = real_images.shape[0]
            d_losses.update(d_loss.item(), batch_size)
            g_losses.update(g_loss.item(), batch_size)

            # Record the total time of training a batch
            batch_time.update(time.time() - end)
            end = time.time()

            # Output training log information once
            if batch_index % config["TRAIN"]["PRINT_FREQ"] == 0:
                # write training log
                iters = batch_index + config["TRAIN"]["HYP"]["EPOCHS"] * batches
                writer.add_scalar("Train/D_Loss", d_loss.item(), iters)
                writer.add_scalar("Train/D(GT)_Loss", d_loss_real.item(), iters)
                writer.add_scalar("Train/D(G(z))_Loss", d_loss_fake.item(), iters)
                writer.add_scalar("Train/G_Loss", g_loss.item(), iters)
                writer.add_scalar("Train/D_Classification_Loss", d_loss_class.item(), iters)
                writer.add_scalar("Train/D_GP_Loss", d_loss_gp.item(), iters)
                writer.add_scalar("Train/G_Classification_Loss", g_loss_class.item(), iters)
                writer.add_scalar("Train/G_REC_Loss", g_loss_rec.item(), iters)
                progress.display(batch_index)

            # Preload the next batch of data
            batch_data = train_data_prefetcher.next()

            # After training a batch of data, add 1 to the number of data batches to ensure that the terminal prints data normally
            batch_index += 1


if __name__ == "__main__":
    main()
