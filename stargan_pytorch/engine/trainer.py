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
import time
from pathlib import Path
from typing import Dict, Union

import torch
from torch import nn, optim, Tensor
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image

from stargan_pytorch.data.celeba import CelebADataset
from stargan_pytorch.data.prefetcher import CPUPrefetcher, CUDAPrefetcher
from stargan_pytorch.models.losses import GradientPenaltyLoss
from stargan_pytorch.models.path_discriminator import path_discriminator
from stargan_pytorch.models.stargan import generator
from stargan_pytorch.models.utils import load_resume_state_dict, load_state_dict
from stargan_pytorch.utils import AverageMeter, ProgressMeter, create_labels, denorm


class Trainer:
    def __init__(
            self,
            config: Dict,
            device: torch.device,
            save_weights_dir: Union[str, Path],
            save_visuals_dir: Union[str, Path],
            tblogger: SummaryWriter,
    ) -> None:
        self.config = config
        self.device = device
        self.save_weights_dir = save_weights_dir
        self.save_visuals_dir = save_visuals_dir
        self.tblogger = tblogger

        self.start_epoch = 0
        self.class_loss_weight = None
        self.gp_loss_weight = None
        self.rec_loss_weight = None

        self.g_model, self.ema_g_model, self.d_model = self.build_model()
        self.class_criterion, self.gp_criterion, self.rec_criterion = self.define_loss()
        self.g_optim, self.d_optim = self.define_optim()
        self.g_scheduler, self.d_scheduler = self.define_scheduler()
        self.dataloader = self.load_datasets()
        self.batches = len(self.dataloader)

        # For training visualization, select a fixed batch of data
        self.fixed_data = self.dataloader.next()
        self.imgs_fixed = self.fixed_data["img"]
        self.label_fixed = self.fixed_data["label"]
        self.imgs_fixed = self.imgs_fixed.to(self.device)
        self.label_fixed = self.label_fixed.to(self.device)
        self.label_fixed_list = create_labels(self.label_fixed,
                                              self.config["MODEL"]["G"]["C_DIM"],
                                              "CelebA",  # Only support CelebA dataset
                                              self.config["DATASETS"]["SELECTED_ATTRS"])

    def build_model(self) -> tuple:
        """Build the generator, discriminator and exponential average generator models

        Returns:
            tuple: generator, exponential average generator and discriminator models
        """

        g_model = generator(c_dim=self.config["MODEL"]["G"]["C_DIM"])
        d_model = path_discriminator(img_size=self.config["MODEL"]["D"]["IMG_SIZE"], c_dim=self.config["MODEL"]["D"]["C_DIM"])

        g_model = g_model.to(self.device)
        d_model = d_model.to(self.device)

        # Generate an exponential average models based on the generator to stabilize models training
        ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: 0.001 * averaged_model_parameter + 0.999 * model_parameter
        ema_g_model = AveragedModel(g_model, self.device, ema_avg_fn)

        # Compile model
        if self.config["MODEL"]["G"]["COMPILED"]:
            g_model = torch.compile(g_model)
            ema_g_model = torch.compile(ema_g_model)
        elif self.config["MODEL"]["D"]["COMPILED"]:
            d_model = torch.compile(d_model)

        return g_model, ema_g_model, d_model

    def define_loss(self) -> tuple:
        """Define the loss function

        Returns:
            tuple: classification loss, gradient penalty loss and reconstruction loss
        """

        class_criterion = nn.BCEWithLogitsLoss()
        gp_criterion = GradientPenaltyLoss()
        rec_criterion = nn.L1Loss()

        class_criterion = class_criterion.to(self.device)
        gp_criterion = gp_criterion.to(self.device)
        rec_criterion = rec_criterion.to(self.device)

        # Define loss function weights
        self.class_loss_weight = torch.Tensor(self.config["TRAIN"]["LOSSES"]["CLASS_LOSS"]["WEIGHT"]).to(self.device)
        self.gp_loss_weight = torch.Tensor(self.config["TRAIN"]["LOSSES"]["GP_LOSS"]["WEIGHT"]).to(self.device)
        self.rec_loss_weight = torch.Tensor(self.config["TRAIN"]["LOSSES"]["REC_LOSS"]["WEIGHT"]).to(self.device)

        return class_criterion, gp_criterion, rec_criterion

    def define_optim(self) -> tuple:
        """Define the optimizer

        Returns:
            torch.optim.Optimizer: generator optimizer, discriminator optimizer
        """

        self.g_optim = optim.Adam(
            self.g_model.parameters(),
            self.config["TRAIN"]["OPTIM"]["G"]["LR"],
            (self.config["TRAIN"]["OPTIM"]["G"]["BETA1"], self.config["TRAIN"]["OPTIM"]["G"]["BETA2"]),
            weight_decay=self.config["TRAIN"]["OPTIM"]["G"]["WEIGHT_DECAY"],
        )
        self.d_optim = optim.Adam(
            self.d_model.parameters(),
            self.config["TRAIN"]["OPTIM"]["D"]["LR"],
            (self.config["TRAIN"]["OPTIM"]["D"]["BETA1"], self.config["TRAIN"]["OPTIM"]["D"]["BETA2"]),
            weight_decay=self.config["TRAIN"]["OPTIM"]["D"]["WEIGHT_DECAY"],
        )

        return self.g_optim, self.d_optim

    def define_scheduler(self) -> tuple:
        """Define the scheduler

        Returns:
            torch.optim.lr_scheduler: generator scheduler, discriminator scheduler
        """

        g_scheduler = optim.lr_scheduler.StepLR(
            self.g_optim,
            self.config["TRAIN"]["LR_SCHEDULER"]["G"]["STEP_SIZE"],
            self.config["TRAIN"]["LR_SCHEDULER"]["G"]["GAMMA"],
        )
        d_scheduler = optim.lr_scheduler.StepLR(
            self.d_optim,
            self.config["TRAIN"]["LR_SCHEDULER"]["D"]["STEP_SIZE"],
            self.config["TRAIN"]["LR_SCHEDULER"]["D"]["GAMMA"],
        )

        return g_scheduler, d_scheduler

    def load_datasets(self) -> CPUPrefetcher or CUDAPrefetcher:
        """Load the dataset and generate the dataset iterator

        Returns:
            DataLoader: dataset iterator
        """

        transform = transforms.Compose([
            transforms.CenterCrop(self.config["DATASETS"]["CROP_IMG_SIZE"]),
            transforms.Resize(self.config["DATASETS"]["RESIZE_IMG_SIZE"]),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        # Load the train dataset
        datasets = CelebADataset(
            self.config["DATASETS"]["IMGS_DIR"],
            transform,
            self.config["DATASETS"]["ATTR_PATH"],
            self.config["DATASETS"]["SELECTED_ATTRS"],
        )
        # generate dataset iterator
        dataloader = DataLoader(datasets,
                                batch_size=self.config["TRAIN"]["HYP"]["IMGS_PER_BATCH"],
                                shuffle=True,
                                num_workers=4,
                                pin_memory=True,
                                drop_last=True,
                                persistent_workers=True)

        if self.device == "cuda":
            # Replace the data set iterator with CUDA to speed up
            dataloader = CUDAPrefetcher(dataloader, self.device)
        else:
            dataloader = CPUPrefetcher(dataloader)

        return dataloader

    def update_g(self, imgs: Tensor, src_label: Tensor, trg_label: Tensor):
        # Disable discriminator backpropagation during generator training
        for d_parameters in self.d_model.parameters():
            d_parameters.requires_grad = False

        # Initialize the generator model gradient
        self.g_model.zero_grad(set_to_none=True)

        # Calculate the perceptual loss of the generator, mainly including pixel loss, feature loss and confrontation loss
        fake_imgs = self.g_model(imgs, trg_label)
        fake_output, fake_class = self.d_model(fake_imgs)
        g_loss_fake = - torch.mean(fake_output)
        g_loss_class = self.class_criterion(fake_class, trg_label)
        g_loss_class = torch.sum(torch.mul(self.class_loss_weight, g_loss_class))

        # Target-to-original domain.
        rec_imgs = self.g_model(fake_imgs, src_label)
        g_loss_rec = self.rec_criterion(rec_imgs, imgs)
        g_loss_rec = torch.sum(torch.mul(self.rec_loss_weight, g_loss_rec))

        # Compute the generator total loss value
        g_loss = g_loss_fake + g_loss_rec + g_loss_class
        # Backpropagation generator loss on generated samples
        g_loss.backward()
        # update generator model weights
        self.g_optim.step()
        # end training generator model

        # update exponentially averaged model weights
        self.ema_g_model.update_parameters(self.g_model)

        return g_loss, g_loss_fake, g_loss_class, g_loss_rec

    def update_d(self, imgs: Tensor, src_label: Tensor, trg_label: Tensor):
        # Start training the discriminator model
        # During discriminator model training, enable discriminator model backpropagation
        for d_parameters in self.d_model.parameters():
            d_parameters.requires_grad = True

        # Initialize the discriminator model gradient
        self.d_model.zero_grad(set_to_none=True)

        # Calculate the classification score of the discriminator model on real samples
        real_output, real_class = self.d_model(imgs)
        d_loss_real = - torch.mean(real_output)
        d_loss_class = self.class_criterion(real_class, src_label)
        d_loss_class = torch.sum(torch.mul(self.class_loss_weight, d_loss_class))

        # Calculate the classification score of the generated samples by the discriminator model
        fake_imgs = self.g_model(imgs, trg_label)
        fake_output, _ = self.d_model(fake_imgs.detach())
        d_loss_fake = torch.mean(fake_output)

        # Compute loss for gradient penalty.
        alpha = torch.rand(imgs.size(0), 1, 1, 1, device=self.device)
        x_hat = (alpha * imgs.data + (1 - alpha) * fake_imgs.data).requires_grad_(True)
        output, _ = self.d_model(x_hat)
        d_loss_gp = self.gp_criterion(output, x_hat)
        d_loss_gp = torch.sum(torch.mul(self.gp_loss_weight, d_loss_gp))

        # Compute the discriminator total loss value
        d_loss = d_loss_real + d_loss_fake + d_loss_class + d_loss_gp
        # backpropagate discriminator loss on generated samples
        d_loss.backward()
        # Update discriminator model weights
        self.d_optim.step()
        # end training discriminator model

        return d_loss, d_loss_real, d_loss_fake, d_loss_class, d_loss_gp

    def load_checkpoint(self) -> None:
        """Load the checkpoint"""

        pretrained_g_model_weights_path = self.config["TRAIN"]["CHECKPOINT"]["G"]["PRETRAINED_MODEL_WEIGHTS_PATH"]
        pretrained_d_model_weights_path = self.config["TRAIN"]["CHECKPOINT"]["D"]["PRETRAINED_MODEL_WEIGHTS_PATH"]
        resume_g_model_weights_path = self.config["TRAIN"]["CHECKPOINT"]["G"]["RESUME_MODEL_WEIGHTS_PATH"]
        resume_d_model_weights_path = self.config["TRAIN"]["CHECKPOINT"]["D"]["RESUME_MODEL_WEIGHTS_PATH"]

        # Load pretrained model weights
        if pretrained_g_model_weights_path != "" and os.path.exists(pretrained_g_model_weights_path):
            print(f"Load checkpoint from '{pretrained_g_model_weights_path}'")
            state_dict = torch.load(pretrained_g_model_weights_path, map_location=self.device)["state_dict"]
            self.g_model = load_state_dict(self.g_model, state_dict)
        if pretrained_d_model_weights_path != "" and os.path.exists(pretrained_d_model_weights_path):
            print(f"Load checkpoint from '{pretrained_d_model_weights_path}'")
            state_dict = torch.load(pretrained_d_model_weights_path, map_location=self.device)["state_dict"]
            self.d_model = load_state_dict(self.d_model, state_dict)

        # Load resume model weights
        if resume_g_model_weights_path != "":
            print(f"Load resume checkpoint from '{resume_g_model_weights_path}'")
            self.start_epoch, self.g_model, self.ema_g_model, self.g_optim, self.d_scheduler = load_resume_state_dict(self.g_model,
                                                                                                                      self.ema_g_model,
                                                                                                                      resume_g_model_weights_path)
        if resume_d_model_weights_path != "":
            print(f"Load resume checkpoint from '{resume_d_model_weights_path}'")
            self.start_epoch, self.d_model, _, self.d_optim, self.d_scheduler = load_resume_state_dict(self.d_model,
                                                                                                       None,
                                                                                                       resume_d_model_weights_path)

    def visual_on_iters(self, iters: int):
        with torch.no_grad():
            img_fake_list = [self.imgs_fixed]
            for label_fixed in self.label_fixed_list:
                img_fake_list.append(self.g_model(self.imgs_fixed, label_fixed))
            img_concat = torch.cat(img_fake_list, dim=3)
            save_sample_path = os.path.join(self.save_visuals_dir, f"iter-{iters:06d}.jpg")
            save_image(denorm(img_concat.data.cpu()), save_sample_path, nrow=1, padding=0)

    def save_checkpoint(self, epoch: int) -> None:
        # Automatically save models weights
        is_last = (epoch + 1) == self.config["TRAIN"]["HYP"]["EPOCHS"]

        g_state_dict = {
            "epoch": epoch + 1,
            "state_dict": self.g_model.state_dict(),
            "ema_state_dict": self.ema_g_model.state_dict(),
            "optimizer": self.g_optim,
            "scheduler": self.g_scheduler,
        }
        d_state_dict = {
            "epoch": epoch + 1,
            "state_dict": self.d_model.state_dict(),
            "ema_state_dict": None,
            "optimizer": self.d_optim,
            "scheduler": self.d_scheduler
        }

        if self.config["TRAIN"]["SAVE_EVERY_EPOCH"] and not is_last:
            g_weights_path = os.path.join(self.save_weights_dir, f"g_epoch_{epoch}.pth.tar")
            d_weights_path = os.path.join(self.save_weights_dir, f"d_epoch_{epoch}.pth.tar")
            torch.save(g_state_dict, g_weights_path)
            torch.save(d_state_dict, d_weights_path)
        else:
            g_weights_path = os.path.join(self.save_weights_dir, f"g_last.pth.tar")
            d_weights_path = os.path.join(self.save_weights_dir, f"d_last.pth.tar")
            torch.save(g_state_dict, g_weights_path)
            torch.save(d_state_dict, d_weights_path)

    def train_on_epoch(self, epoch: int):
        # The information printed by the progress bar
        global g_loss, g_loss_fake, g_loss_class, g_loss_rec
        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        g_losses = AverageMeter("G Loss", ":6.6f")
        d_losses = AverageMeter("D Loss", ":6.6f")
        progress = ProgressMeter(self.batches,
                                 [batch_time, data_time, g_losses, d_losses],
                                 prefix=f"Epoch: [{epoch}]")

        # Put the generator in training mode
        self.g_model.train()
        self.d_model.train()

        # Initialize data batches
        batch_index = 0
        # Set the dataset iterator pointer to 0
        self.dataloader.reset()
        # Record the start time of training a batch
        end = time.time()
        # load the first batch of data
        batch_data = self.dataloader.next()

        while batch_data is not None:
            # Load batches of data
            imgs = batch_data["img"].to(self.device, non_blocking=True)
            src_label = batch_data["label"].to(self.device, non_blocking=True)

            # Generate target domain labels randomly.
            rand_index = torch.randperm(src_label.size(0))
            trg_label = src_label[rand_index]

            # Record the time to load a batch of data
            data_time.update(time.time() - end)

            # start training the discriminator model
            d_loss, d_loss_real, d_loss_fake, d_loss_class, d_loss_gp = self.update_d(imgs, src_label, trg_label)

            # start training the generator model
            if batch_index % self.config["TRAIN"]["N_CRITIC"] == 0:
                g_loss, g_loss_fake, g_loss_class, g_loss_rec = self.update_g(imgs, src_label, trg_label)

            # record the loss value
            batch_size = imgs.shape[0]
            d_losses.update(d_loss.item(), batch_size)
            g_losses.update(g_loss.item(), batch_size)

            # Record the total time of training a batch
            batch_time.update(time.time() - end)
            end = time.time()

            # Output training log information once
            iters = batch_index + epoch * self.batches
            if batch_index % self.config["TRAIN"]["PRINT_FREQ"] == 0:
                # write training log
                self.tblogger.add_scalar("Train/D_Loss", d_loss.item(), iters)
                self.tblogger.add_scalar("Train/D(GT)_Loss", d_loss_real.item(), iters)
                self.tblogger.add_scalar("Train/D(G(z))_Loss", d_loss_fake.item(), iters)
                self.tblogger.add_scalar("Train/D_Class_Loss", d_loss_class.item(), iters)
                self.tblogger.add_scalar("Train/D_GP_Loss", d_loss_gp.item(), iters)
                self.tblogger.add_scalar("Train/G_Loss", g_loss.item(), iters)
                self.tblogger.add_scalar("Train/G(G(z))_Loss", g_loss_fake.item(), iters)
                self.tblogger.add_scalar("Train/G_Class_Loss", g_loss_class.item(), iters)
                self.tblogger.add_scalar("Train/G_REC_Loss", g_loss_rec.item(), iters)
                progress.display(batch_index + 1)

            # Save the generated samples
            if (iters + 1) % self.config["TRAIN"]["VISUAL_FREQ"] == 0:
                self.visual_on_iters(iters + 1)

            # Preload the next batch of data
            batch_data = self.dataloader.next()

            # After training a batch of data, add 1 to the number of data batches to ensure that the terminal prints data normally
            batch_index += 1

    def train(self):
        self.load_checkpoint()

        for epoch in range(self.start_epoch, self.config["TRAIN"]["HYP"]["EPOCHS"]):
            self.train_on_epoch(epoch)

            # Update learning rate scheduler
            self.g_scheduler.step()
            self.d_scheduler.step()

            # Save weights
            self.save_checkpoint(epoch)
