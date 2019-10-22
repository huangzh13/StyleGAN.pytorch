"""
-------------------------------------------------
   File Name:    GAN.py
   Author:       Zhonghao Huang
   Date:         2019/10/17
   Description:
-------------------------------------------------
"""

import os
import datetime
import time
import timeit
import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import AvgPool2d
from torch.optim import Adam

from models.Blocks import GMapping, GSynthesis, DiscriminatorTop, DiscriminatorBlock
from models.CustomLayers import EqualizedConv2d, update_average
import models.Losses as losses


class Generator(nn.Module):
    def __init__(self,
                 truncation_psi=0.7,
                 truncation_cutoff=8,
                 truncation_psi_val=None,
                 truncation_cutoff_val=None,
                 dlatent_avg_beta=0.995,
                 style_mixing_prob=0.9,
                 **kwargs):
        """
        # Style-based generator used in the StyleGAN paper.
        # Composed of two sub-networks (G_mapping and G_synthesis).
        :param truncation_psi: Style strength multiplier for the truncation trick. None = disable.
        :param truncation_cutoff: Number of layers for which to apply the truncation trick. None = disable.
        :param truncation_psi_val: Value for truncation_psi to use during validation.
        :param truncation_cutoff_val: Value for truncation_cutoff to use during validation.
        :param dlatent_avg_beta: Decay for tracking the moving average of W during training. None = disable.
        :param style_mixing_prob: Probability of mixing styles during training. None = disable.
        :param kwargs: Arguments for sub-networks (G_mapping and G_synthesis).
        """

        super(Generator, self).__init__()

        # Setup components.
        # TODO

        self.g_mapping = GMapping(**kwargs)
        self.g_synthesis = GSynthesis(**kwargs)

        # Update moving average of W.
        # TODO

    def forward(self, latents_in, labels_in=None):
        """
        :param latents_in: First input: Latent vectors (Z) [mini_batch, latent_size].
        :param labels_in: Second input: Conditioning labels [mini_batch, label_size].
        :return:
        """

        dlatents_in = self.g_mapping(latents_in)

        # Perform style mixing regularization.
        # TODO
        # Apply truncation trick.
        # TODO

        fake_images = self.g_synthesis(dlatents_in)

        return fake_images


class Discriminator(nn.Module):
    def __init__(self, num_channels=1, resolution=32, fmap_base=8192, fmap_decay=1.0, fmap_max=512,
                 nonlinearity='lrelu', use_wscale=True, mbstd_group_size=4, mbstd_num_features=1,
                 blur_filter=None, structure='liner', **kwargs):
        """
        Discriminator used in the StyleGAN paper.
        :param num_channels: Number of input color channels. Overridden based on dataset.
        :param resolution: Input resolution. Overridden based on dataset.
        # label_size=0,  # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
        :param fmap_base: Overall multiplier for the number of feature maps.
        :param fmap_decay: log2 feature map reduction when doubling the resolution.
        :param fmap_max: Maximum number of feature maps in any layer.
        :param nonlinearity: Activation function: 'relu', 'lrelu'
        :param use_wscale: Enable equalized learning rate?
        :param mbstd_group_size: Group size for the mini_batch standard deviation layer, 0 = disable.
        :param mbstd_num_features: Number of features for the mini_batch standard deviation layer.
        :param blur_filter: Low-pass filter to apply when resampling activations. None = no filtering.
        :param structure: 'fixed' = no progressive growing, 'linear' = human-readable
        :param kwargs: Ignore unrecognized keyword args.
        """
        super(Discriminator, self).__init__()

        self.mbstd_num_features = mbstd_num_features
        self.mbstd_group_size = mbstd_group_size
        self.structure = structure
        if blur_filter is None:
            blur_filter = [1, 2, 1]

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]

        # Building the final block.
        final_block = [('4x4', DiscriminatorTop(self.mbstd_group_size, self.mbstd_num_features,
                                                in_channels=nf(2), intermediate_channels=nf(2),
                                                gain=gain, use_wscale=use_wscale, activation_layer=act))]
        self.top = nn.Sequential(OrderedDict(final_block))

        # create the fromRGB layers for various inputs:
        from_rgb = [EqualizedConv2d(num_channels, nf(resolution_log2 - 1), kernel_size=1,
                                    gain=gain, use_wscale=use_wscale)]

        # create the remaining layers
        blocks = []
        for res in range(3, resolution_log2 + 1):
            name = '{s}x{s}'.format(s=2 ** res)
            blocks.append((name, DiscriminatorBlock(nf(res - 1), nf(res - 2),
                                                    gain=gain, use_wscale=use_wscale, activation_layer=act)))
            # create the fromRGB layers for various inputs:
            from_rgb.append(EqualizedConv2d(num_channels, nf(res - 1), kernel_size=1,
                                            gain=gain, use_wscale=use_wscale))

        self.blocks = nn.Sequential(OrderedDict(blocks))
        self.from_rgb = nn.ModuleList(from_rgb)

        # register the temporary downSampler
        self.temporaryDownsampler = AvgPool2d(2)

    def forward(self, images_in, depth=None, alpha=1., labels_in=None):
        """
        :param images_in: First input: Images [mini_batch, channel, height, width].
        :param labels_in: Second input: Labels [mini_batch, label_size].
        :param depth: current height of operation (Progressive GAN)
        :param alpha: current value of alpha for fade-in
        :return:
        """

        assert depth < self.depth, "Requested output depth cannot be produced"

        if self.structure == 'fixed':
            # TODO
            scores_out = images_in
        elif self.structure == 'linear':
            if depth > 0:
                residual = self.from_rgb[depth - 1](self.temporaryDownsampler(images_in))
                straight = self.blocks[depth - 1](self.from_rgb[depth](images_in))
                y = (alpha * straight) + ((1 - alpha) * residual)

                for block in reversed(self.blocks[:depth - 1]):
                    y = block(y)
            else:
                y = self.from_rgb[0](images_in)

            scores_out = self.top(y)
        else:
            raise KeyError("Unknown structure: ", self.structure)

        return scores_out


class StyleGAN:
    """ Wrapper around the Generator and the Discriminator """

    def __init__(self, g_args, d_args, g_opt_args, d_opt_args, depth=7, latent_size=512, learning_rate=0.001, beta_1=0,
                 beta_2=0.99, eps=1e-8, n_critic=1, use_eql=True,
                 loss="relativistic-hinge", use_ema=True, ema_decay=0.999,
                 device=torch.device("cpu")):
        """
        :param depth:
        :param latent_size:
        :param learning_rate:
        :param beta_1:
        :param beta_2:
        :param eps:
        :param n_critic:
        :param use_eql:
        :param loss:
        :param use_ema:
        :param ema_decay:
        :param device:
        """

        # state of the object
        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # Create the Generator and the Discriminator
        # self.gen = Generator(num_channels=, resolution=, **g_args)
        # self.dis = Discriminator(num_channels=, resolution=, **d_args)

        # if code is to be run on GPU, we can use DataParallel:

        # define the optimizers for the discriminator and generator
        self.__setup_optim(**g_opt_args, **d_opt_args)

        # define the loss function used for training the GAN
        self.loss = self.__setup_loss(loss)

        # Use of ema
        if self.use_ema:
            # create a shadow copy of the generator
            self.gen_shadow = copy.deepcopy(self.gen)
            # updater function:
            self.ema_updater = update_average
            # initialize the gen_shadow weights equal to the weights of gen
            self.ema_updater(self.gen_shadow, self.gen, beta=0)

    def __setup_optim(self, learning_rate, beta_1, beta_2, eps):
        self.gen_optim = Adam(self.gen.parameters(), lr=learning_rate, betas=(beta_1, beta_2), eps=eps)
        self.dis_optim = Adam(self.dis.parameters(), lr=learning_rate, betas=(beta_1, beta_2), eps=eps)

    def __setup_loss(self, loss):
        if isinstance(loss, str):
            loss = loss.lower()  # lowercase the string

            if loss == "standard-gan":
                loss = losses.StandardGAN(self.dis)

            elif loss == "hinge":
                loss = losses.HingeGAN(self.dis)

            elif loss == "relativistic-hinge":
                loss = losses.RelativisticAverageHingeGAN(self.dis)

            else:
                raise ValueError("Unknown loss function requested")

        elif not isinstance(loss, losses.GANLoss):
            raise ValueError("loss is neither an instance of GANLoss nor a string")

        return loss

    def __progressive_down_sampling(self, real_batch, depth, alpha):
        """
        private helper for down_sampling the original images in order to facilitate the
        progressive growing of the layers.
        :param real_batch: batch of real samples
        :param depth: depth at which training is going on
        :param alpha: current value of the fade-in alpha
        :return: real_samples => modified real batch of samples
        """

        from torch.nn import AvgPool2d
        from torch.nn.functional import interpolate

        # down_sample the real_batch for the given depth
        down_sample_factor = int(np.power(2, self.depth - depth - 1))
        prior_down_sample_factor = max(int(np.power(2, self.depth - depth)), 0)

        ds_real_samples = AvgPool2d(down_sample_factor)(real_batch)

        if depth > 0:
            prior_ds_real_samples = interpolate(AvgPool2d(prior_down_sample_factor)(real_batch), scale_factor=2)
        else:
            prior_ds_real_samples = ds_real_samples

        # real samples are a combination of ds_real_samples and prior_ds_real_samples
        real_samples = (alpha * ds_real_samples) + ((1 - alpha) * prior_ds_real_samples)

        # return the so computed real_samples
        return real_samples

    def optimize_discriminator(self, noise, real_batch, depth, alpha):
        """
        performs one step of weight update on discriminator using the batch of data
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
        :param depth: current depth of optimization
        :param alpha: current alpha for fade-in
        :return: current loss (Wasserstein loss)
        """

        real_samples = self.__progressive_down_sampling(real_batch, depth, alpha)

        loss_val = 0
        for _ in range(self.n_critic):
            # generate a batch of samples
            fake_samples = self.gen(noise, depth, alpha).detach()

            loss = self.loss.dis_loss(real_samples, fake_samples, depth, alpha)

            # optimize discriminator
            self.dis_optim.zero_grad()
            loss.backward()
            self.dis_optim.step()

            loss_val += loss.item()

        return loss_val / self.n_critic

    def optimize_generator(self, noise, real_batch, depth, alpha):
        """
        performs one step of weight update on generator for the given batch_size

        :param noise: input random noise required for generating samples
        :param real_batch: batch of real samples
        :param depth: depth of the network at which optimization is done
        :param alpha: value of alpha for fade-in effect
        :return: current loss (Wasserstein estimate)
        """

    def train(self, dataset, epochs, batch_sizes, fade_in_percentage, num_samples=16,
              start_depth=0, num_workers=3, feedback_factor=100, checkpoint_factor=1,
              log_dir="./models/", sample_dir="./samples/", save_dir="./models/"):
        """
        Utility method for training the ProGAN. Note that you don't have to necessarily use this
        you can use the optimize_generator and optimize_discriminator for your own training routine.

        :param dataset: object of the dataset used for training.
                        Note that this is not the data loader (we create data loader in this method
                        since the batch_sizes for resolutions can be different)
        :param epochs: list of number of epochs to train the network for every resolution
        :param batch_sizes: list of batch_sizes for every resolution
        :param fade_in_percentage: list of percentages of epochs per resolution
                                   used for fading in the new layer
                                   not used for first resolution, but dummy value still needed.
        :param num_samples: number of samples generated in sample_sheet. def=36
        :param start_depth: start training from this depth. def=0
        :param num_workers: number of workers for reading the data. def=3
        :param feedback_factor: number of logs per epoch. def=100
        :param log_dir: directory for saving the loss logs. def="./models/"
        :param sample_dir: directory for saving the generated samples. def="./samples/"
        :param checkpoint_factor: save model after these many epochs.
                                  Note that only one model is stored per resolution.
                                  during one resolution, the checkpoint will be updated (Rewritten)
                                  according to this factor.
        :param save_dir: directory for saving the models (.pth files)
        :return: None (Writes multiple files to disk)
        """

        assert self.depth == len(batch_sizes), "batch_sizes not compatible with depth"

        # turn the generator and discriminator into train mode
        self.gen.train()
        self.dis.train()
        if self.use_ema:
            self.gen_shadow.train()

        # create a global time counter
        global_time = time.time()

        # create fixed_input for debugging
        fixed_input = torch.randn(num_samples, self.latent_size).to(self.device)

        print("Starting the training process ... ")
        for current_depth in range(start_depth, self.depth):
            print("\n\nCurrently working on Depth: ", current_depth)
            # Choose training parameters and configure training ops.

            current_res = np.power(2, current_depth + 2)
            print("Current resolution: %d x %d" % (current_res, current_res))

            data = get_data_loader(dataset, batch_sizes[current_depth], num_workers)
            ticker = 1

            for epoch in range(1, epochs[current_depth] + 1):
                start = timeit.default_timer()  # record time at the start of epoch

                print("\nEpoch: %d" % epoch)
                total_batches = len(iter(data))

                fader_point = int((fade_in_percentage[current_depth] / 100)
                                  * epochs[current_depth] * total_batches)

                step = 0  # counter for number of iterations

                for (i, batch) in enumerate(data, 1):
                    # calculate the alpha for fading in the layers
                    alpha = ticker / fader_point if ticker <= fader_point else 1

                    # extract current batch of data for training
                    images = batch.to(self.device)

                    gan_input = torch.randn(images.shape[0], self.latent_size).to(self.device)

                    # optimize the discriminator:
                    dis_loss = self.optimize_discriminator(gan_input, images, current_depth, alpha)

                    # optimize the generator:
                    gen_loss = self.optimize_generator(gan_input, images, current_depth, alpha)

                    # provide a loss feedback
                    if i % int(total_batches / feedback_factor) == 0 or i == 1:
                        elapsed = time.time() - global_time
                        elapsed = str(datetime.timedelta(seconds=elapsed))
                        print("Elapsed: [%s]  batch: %d  d_loss: %f  g_loss: %f"
                              % (elapsed, i, dis_loss, gen_loss))

                        # also write the losses to the log file:
                        os.makedirs(log_dir, exist_ok=True)
                        log_file = os.path.join(log_dir, "loss_" + str(current_depth) + ".log")
                        with open(log_file, "a") as log:
                            log.write(str(step) + "\t" + str(dis_loss) + "\t" + str(gen_loss) + "\n")

                        # create a grid of samples and save it
                        os.makedirs(sample_dir, exist_ok=True)
                        gen_img_file = os.path.join(sample_dir, "gen_" + str(current_depth) +
                                                    "_" + str(epoch) + "_" + str(i) + ".png")


if __name__ == '__main__':
    net = StyleGAN()

    print('Done.')
