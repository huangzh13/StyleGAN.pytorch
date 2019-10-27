"""
-------------------------------------------------
   File Name:    GAN.py
   Author:       Zhonghao Huang
   Date:         2019/10/17
   Description:  Modified from: https://github.com/akanimax/pro_gan_pytorch
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
from torch.nn.functional import interpolate

from data import get_data_loader
from models import update_average
from models.Blocks import DiscriminatorTop, DiscriminatorBlock, InputBlock, GSynthesisBlock
from models.CustomLayers import EqualizedConv2d, PixelNormLayer, EqualizedLinear
import models.Losses as Losses


class GMapping(nn.Module):
    """
        Mapping network used in the StyleGAN paper.
    """

    def __init__(self,
                 latent_size=512,  # Latent vector(Z) dimensionality.
                 # label_size=0,  # Label dimensionality, 0 if no labels.
                 dlatent_size=512,  # Disentangled latent (W) dimensionality.
                 dlatent_broadcast=None,
                 mapping_layers=8,  # Number of mapping layers.
                 mapping_fmaps=512,  # Number of activations in the mapping layers.
                 mapping_lrmul=0.01,  # Learning rate multiplier for the mapping layers.
                 mapping_nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu'.
                 use_wscale=True,  # Enable equalized learning rate?
                 normalize_latents=True,  # Normalize latent vectors (Z) before feeding them to the mapping layers?
                 # dtype='float32',  # Data type to use for activations and outputs.
                 **kwargs):  # Ignore unrecognized keyword args.

        super().__init__()

        self.latent_size = latent_size
        self.mapping_fmaps = mapping_fmaps
        self.dlatent_size = dlatent_size
        # Output disentangled latent (W) as [mini_batch, dlatent_size] or [mini_batch, dlatent_broadcast, dlatent_size].
        self.dlatent_broadcast = dlatent_broadcast

        # Activation function.
        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[mapping_nonlinearity]

        # Embed labels and concatenate them with latents.
        # TODO

        layers = []
        # Normalize latents.
        if normalize_latents:
            layers.append(('pixel_norm', PixelNormLayer()))

        # Mapping layers. (apply_bias?)
        layers.append(('dense0', EqualizedLinear(self.latent_size, self.mapping_fmaps,
                                                 gain=gain, lrmul=mapping_lrmul, use_wscale=use_wscale)))
        layers.append(('dense0_act', act))
        for layer_idx in range(1, mapping_layers):
            fmaps_in = self.mapping_fmaps
            fmaps_out = self.dlatent_size if layer_idx == mapping_layers - 1 else self.mapping_fmaps
            layers.append(
                ('dense{:d}'.format(layer_idx),
                 EqualizedLinear(fmaps_in, fmaps_out, gain=gain, lrmul=mapping_lrmul, use_wscale=use_wscale)))
            layers.append(('dense{:d}_act'.format(layer_idx), act))

        # Output.
        self.map = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        # First input: Latent vectors (Z) [mini_batch, latent_size].
        x = self.map(x)
        # Broadcast -> batch_size * dlatent_broadcast * dlatent_size
        if self.dlatent_broadcast is not None:
            x = x.unsqueeze(1).expand(-1, self.dlatent_broadcast, -1)
        return x


class GSynthesis(nn.Module):
    """
        Synthesis network used in the StyleGAN paper.
    """

    def __init__(self,
                 dlatent_size=512,  # Disentangled latent (W) dimensionality.
                 num_channels=3,  # Number of output color channels.
                 resolution=1024,  # Output resolution.
                 fmap_base=8192,  # Overall multiplier for the number of feature maps.
                 fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
                 fmap_max=512,  # Maximum number of feature maps in any layer.
                 use_styles=True,  # Enable style inputs?
                 const_input_layer=True,  # First layer is a learned constant?
                 use_noise=True,  # Enable noise inputs?
                 nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu'
                 use_wscale=True,  # Enable equalized learning rate?
                 use_pixel_norm=False,  # Enable pixelwise feature vector normalization?
                 use_instance_norm=True,  # Enable instance normalization?
                 blur_filter=None,  # Low-pass filter to apply when resampling activations. None = no filtering.
                 structure='linear',
                 **kwargs):  # Ignore unrecognized keyword args.

        super().__init__()

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.structure = structure

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        self.depth = resolution_log2 - 1

        self.num_layers = resolution_log2 * 2 - 2
        self.num_styles = self.num_layers if use_styles else 1

        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]

        # Early layers.
        self.init_block = InputBlock(nf(1), dlatent_size, const_input_layer, gain, use_wscale,
                                     use_noise, use_pixel_norm, use_instance_norm, use_styles, act)
        # create the ToRGB layers for various outputs
        rgb_converters = [EqualizedConv2d(nf(1), num_channels, 1, gain=1, use_wscale=use_wscale)]

        # Building blocks for remaining layers.
        blocks = []
        for res in range(3, resolution_log2 + 1):
            last_channels = nf(res - 2)
            channels = nf(res - 1)
            # name = '{s}x{s}'.format(s=2 ** res)
            blocks.append(GSynthesisBlock(last_channels, channels, blur_filter, dlatent_size, gain, use_wscale,
                                          use_noise, use_pixel_norm, use_instance_norm, use_styles, act))
            rgb_converters.append(EqualizedConv2d(channels, num_channels, 1, gain=1, use_wscale=use_wscale))

        self.blocks = nn.ModuleList(blocks)
        self.to_rgb = nn.ModuleList(rgb_converters)

        # register the temporary upsampler
        self.temporaryUpsampler = lambda x: interpolate(x, scale_factor=2)

    def forward(self, dlatents_in, depth=0, alpha=0., labels_in=None):
        """
            forward pass of the Generator
            :param dlatents_in: Input: Disentangled latents (W) [mini_batch, num_layers, dlatent_size].
            :param labels_in:
            :param depth: current depth from where output is required
            :param alpha: value of alpha for fade-in effect
            :return: y => output
        """

        assert depth < self.depth, "Requested output depth cannot be produced"

        if self.structure == 'fixed':
            x = self.init_block(dlatents_in[:, 0:2])
            for i, block in enumerate(self.blocks):
                x = block(x, dlatents_in[:, 2 * (i + 1):2 * (i + 2)])
            images_out = self.to_rgb[-1](x)
        elif self.structure == 'linear':
            x = self.init_block(dlatents_in[:, 0:2])

            if depth > 0:
                for i, block in enumerate(self.blocks[:depth - 1]):
                    x = block(x, dlatents_in[:, 2 * (i + 1):2 * (i + 2)])

                residual = self.to_rgb[depth - 1](self.temporaryUpsampler(x))
                straight = self.to_rgb[depth](self.blocks[depth - 1](x, dlatents_in[:, 2 * depth:2 * (depth + 1)]))

                images_out = (alpha * straight) + ((1 - alpha) * residual)
            else:
                images_out = self.to_rgb[0](x)
        else:
            raise KeyError("Unknown structure: ", self.structure)

        return images_out


class Generator(nn.Module):
    def __init__(self,
                 resolution,
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
        num_layers = (int(np.log2(resolution)) - 1) * 2
        self.g_mapping = GMapping(dlatent_broadcast=num_layers, **kwargs)
        self.g_synthesis = GSynthesis(resolution=resolution, **kwargs)

        # Update moving average of W.
        # TODO

    def forward(self, latents_in, depth, alpha, labels_in=None):
        """
        :param latents_in: First input: Latent vectors (Z) [mini_batch, latent_size].
        :param depth:
        :param alpha:
        :param labels_in: Second input: Conditioning labels [mini_batch, label_size].
        :return:
        """

        dlatents_in = self.g_mapping(latents_in)

        # Perform style mixing regularization.
        # TODO
        # Apply truncation trick.
        # TODO

        fake_images = self.g_synthesis(dlatents_in, depth, alpha)

        return fake_images


class Discriminator(nn.Module):
    def __init__(self, resolution, num_channels=3, fmap_base=8192, fmap_decay=1.0, fmap_max=512,
                 nonlinearity='lrelu', use_wscale=True, mbstd_group_size=4, mbstd_num_features=1,
                 blur_filter=None, structure='linear', **kwargs):
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

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        self.mbstd_num_features = mbstd_num_features
        self.mbstd_group_size = mbstd_group_size
        self.structure = structure
        if blur_filter is None:
            blur_filter = [1, 2, 1]

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
        self.depth = resolution_log2 - 1

        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]

        # create the remaining layers
        blocks = []
        from_rgb = []
        for res in range(resolution_log2, 2, -1):
            # name = '{s}x{s}'.format(s=2 ** res)
            blocks.append(DiscriminatorBlock(nf(res - 1), nf(res - 2),
                                             gain=gain, use_wscale=use_wscale, activation_layer=act))
            # create the fromRGB layers for various inputs:
            from_rgb.append(EqualizedConv2d(num_channels, nf(res - 1), kernel_size=1,
                                            gain=gain, use_wscale=use_wscale))
        self.blocks = nn.ModuleList(blocks)

        # Building the final block.
        self.final_block = DiscriminatorTop(self.mbstd_group_size, self.mbstd_num_features,
                                            in_channels=nf(2), intermediate_channels=nf(2),
                                            gain=gain, use_wscale=use_wscale, activation_layer=act)
        from_rgb.append(EqualizedConv2d(num_channels, nf(2), kernel_size=1,
                                        gain=gain, use_wscale=use_wscale))
        self.from_rgb = nn.ModuleList(from_rgb)

        # register the temporary downSampler
        self.temporaryDownsampler = nn.AvgPool2d(2)

    def forward(self, images_in, depth, alpha=1., labels_in=None):
        """
        :param images_in: First input: Images [mini_batch, channel, height, width].
        :param labels_in: Second input: Labels [mini_batch, label_size].
        :param depth: current height of operation (Progressive GAN)
        :param alpha: current value of alpha for fade-in
        :return:
        """

        assert depth < self.depth, "Requested output depth cannot be produced"

        if self.structure == 'fixed':
            x = self.from_rgb[0](images_in)
            for i, block in enumerate(self.blocks):
                x = block(x)
            scores_out = self.final_block(x)
        elif self.structure == 'linear':
            if depth > 0:
                residual = self.from_rgb[self.depth - depth](self.temporaryDownsampler(images_in))
                straight = self.blocks[self.depth - depth - 1](self.from_rgb[self.depth - depth - 1](images_in))

                x = (alpha * straight) + ((1 - alpha) * residual)

                for block in self.blocks[(self.depth - depth):]:
                    x = block(x)
            else:
                x = self.from_rgb[-1](images_in)

            scores_out = self.final_block(x)
        else:
            raise KeyError("Unknown structure: ", self.structure)

        return scores_out


class StyleGAN:
    """ Wrapper around the Generator and the Discriminator """

    def __init__(self, structure, resolution, num_channels,
                 g_args, d_args, g_opt_args, d_opt_args, loss="relativistic-hinge",
                 latent_size=512, d_repeats=1, use_ema=False, ema_decay=0.999,
                 device=torch.device("cpu")):
        """
        :param latent_size:
        :param d_repeats:
        :param loss:
        :param use_ema:
        :param ema_decay:
        :param device:
        """

        # state of the object
        assert structure in ['fixed', 'linear']
        self.structure = structure
        self.depth = int(np.log2(resolution)) - 1

        self.latent_size = latent_size
        self.num_channels = num_channels
        self.d_repeats = d_repeats
        self.device = device

        self.use_ema = use_ema
        self.ema_decay = ema_decay

        # Create the Generator and the Discriminator
        self.gen = Generator(num_channels=self.num_channels,
                             resolution=resolution,
                             structure=self.structure,
                             **g_args).to(self.device)

        self.dis = Discriminator(num_channels=self.num_channels,
                                 resolution=resolution,
                                 structure=self.structure,
                                 **d_args).to(self.device)

        # if code is to be run on GPU, we can use DataParallel:
        # TODO

        # define the optimizers for the discriminator and generator
        self.__setup_gen_optim(**g_opt_args)
        self.__setup_dis_optim(**d_opt_args)

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

    def __setup_gen_optim(self, learning_rate, beta_1, beta_2, eps):
        self.gen_optim = torch.optim.Adam(self.gen.parameters(), lr=learning_rate, betas=(beta_1, beta_2), eps=eps)

    def __setup_dis_optim(self, learning_rate, beta_1, beta_2, eps):
        self.dis_optim = torch.optim.Adam(self.dis.parameters(), lr=learning_rate, betas=(beta_1, beta_2), eps=eps)

    def __setup_loss(self, loss):
        if isinstance(loss, str):
            loss = loss.lower()  # lowercase the string

            if loss == "standard-gan":
                loss = Losses.StandardGAN(self.dis)

            elif loss == "hinge":
                loss = Losses.HingeGAN(self.dis)

            elif loss == "relativistic-hinge":
                loss = Losses.RelativisticAverageHingeGAN(self.dis)

            else:
                raise ValueError("Unknown loss function requested")

        elif not isinstance(loss, Losses.GANLoss):
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

        if self.structure == 'fixed':
            return real_batch

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
        for _ in range(self.d_repeats):
            # generate a batch of samples
            fake_samples = self.gen(noise, depth, alpha).detach()

            loss = self.loss.dis_loss(real_samples, fake_samples, depth, alpha)

            # optimize discriminator
            self.dis_optim.zero_grad()
            loss.backward()
            self.dis_optim.step()

            loss_val += loss.item()

        return loss_val / self.d_repeats

    def optimize_generator(self, noise, real_batch, depth, alpha):
        """
        performs one step of weight update on generator for the given batch_size

        :param noise: input random noise required for generating samples
        :param real_batch: batch of real samples
        :param depth: depth of the network at which optimization is done
        :param alpha: value of alpha for fade-in effect
        :return: current loss (Wasserstein estimate)
        """

        real_samples = self.__progressive_down_sampling(real_batch, depth, alpha)

        # generate fake samples:
        fake_samples = self.gen(noise, depth, alpha)

        # TODO_complete:
        # Change this implementation for making it compatible for relativisticGAN
        loss = self.loss.gen_loss(real_samples, fake_samples, depth, alpha)

        # optimize the generator
        self.gen_optim.zero_grad()
        loss.backward()
        self.gen_optim.step()

        # if use_ema is true, apply ema to the generator parameters
        if self.use_ema:
            self.ema_updater(self.gen_shadow, self.gen, self.ema_decay)

        # return the loss value
        return loss.item()

    @staticmethod
    def create_grid(samples, scale_factor, img_file):
        """
        utility function to create a grid of GAN samples
        :param samples: generated samples for storing
        :param scale_factor: factor for upscaling the image
        :param img_file: name of file to write
        :return: None (saves a file)
        """
        from torchvision.utils import save_image
        from torch.nn.functional import interpolate

        # upsample the image
        if scale_factor > 1:
            samples = interpolate(samples, scale_factor=scale_factor)

        # save the images:
        save_image(samples, img_file, nrow=int(np.sqrt(len(samples))),
                   normalize=True, scale_each=True, pad_value=128, padding=1)

    def train(self, dataset, epochs, batch_sizes, fade_in_percentage, logger, output,
              num_samples=16, start_depth=0, num_workers=3, feedback_factor=100, checkpoint_factor=1):
        """
        Utility method for training the GAN. Note that you don't have to necessarily use this
        you can use the optimize_generator and optimize_discriminator for your own training routine.

        :param output:
        :param logger:
        :param dataset: object of the dataset used for training.
                        Note that this is not the data loader (we create data loader in this method
                        since the batch_sizes for resolutions can be different)
        :param epochs: list of number of epochs to train the network for every resolution
        :param batch_sizes: list of batch_sizes for every resolution
        :param fade_in_percentage: list of percentages of epochs per resolution used for fading in the new layer
                                   not used for first resolution, but dummy value still needed.
        :param num_samples: number of samples generated in sample_sheet. def=36
        :param start_depth: start training from this depth. def=0
        :param num_workers: number of workers for reading the data. def=3
        :param feedback_factor: number of logs per epoch. def=100
        :param checkpoint_factor: save model after these many epochs.
                                  Note that only one model is stored per resolution.
                                  during one resolution, the checkpoint will be updated (Rewritten)
                                  according to this factor.
        :return: None (Writes multiple files to disk)
        """

        assert self.depth <= len(batch_sizes) + 1, "batch_sizes not compatible with depth"
        assert self.depth <= len(batch_sizes) + 1, "batch_sizes not compatible with depth"
        assert self.depth <= len(batch_sizes) + 1, "batch_sizes not compatible with depth"

        # turn the generator and discriminator into train mode
        self.gen.train()
        self.dis.train()
        if self.use_ema:
            self.gen_shadow.train()

        # create a global time counter
        global_time = time.time()

        # create fixed_input for debugging
        fixed_input = torch.randn(num_samples, self.latent_size).to(self.device)

        # config depend on structure
        logger.info("Starting the training process ... \n")
        if self.structure == 'fixed':
            start_depth = self.depth - 1
        step = 1  # counter for number of iterations
        for current_depth in range(start_depth, self.depth):
            current_res = np.power(2, current_depth + 2)
            logger.info("Currently working on depth: %d", current_depth + 1)
            logger.info("Current resolution: %d x %d" % (current_res, current_res))

            ticker = 1

            # Choose training parameters and configure training ops.
            # TODO
            data = get_data_loader(dataset, batch_sizes[current_depth], num_workers)

            for epoch in range(1, epochs[current_depth] + 1):
                start = timeit.default_timer()  # record time at the start of epoch

                logger.info("Epoch: [%d]" % epoch)
                # total_batches = len(iter(data))
                total_batches = len(data)

                fade_point = int((fade_in_percentage[current_depth] / 100)
                                 * epochs[current_depth] * total_batches)

                for (i, batch) in enumerate(data, 1):
                    # calculate the alpha for fading in the layers
                    alpha = ticker / fade_point if ticker <= fade_point else 1

                    # extract current batch of data for training
                    images = batch.to(self.device)
                    gan_input = torch.randn(images.shape[0], self.latent_size).to(self.device)

                    # optimize the discriminator:
                    dis_loss = self.optimize_discriminator(gan_input, images, current_depth, alpha)

                    # optimize the generator:
                    gen_loss = self.optimize_generator(gan_input, images, current_depth, alpha)

                    # provide a loss feedback
                    if i % int(total_batches / feedback_factor + 1) == 0 or i == 1:
                        elapsed = time.time() - global_time
                        elapsed = str(datetime.timedelta(seconds=elapsed)).split('.')[0]
                        logger.info(
                            "Elapsed: [%s] Step: %d  Batch: %d  D_Loss: %f  G_Loss: %f"
                            % (elapsed, step, i, dis_loss, gen_loss))

                        # create a grid of samples and save it
                        os.makedirs(os.path.join(output, 'samples'), exist_ok=True)
                        gen_img_file = os.path.join(output, 'samples', "gen_" + str(current_depth)
                                                    + "_" + str(epoch) + "_" + str(i) + ".png")

                        with torch.no_grad():
                            self.create_grid(
                                samples=self.gen(fixed_input, current_depth, alpha).detach() if not self.use_ema
                                else self.gen_shadow(fixed_input, current_depth, alpha).detach(),
                                scale_factor=int(
                                    np.power(2, self.depth - current_depth - 1)) if self.structure == 'linear' else 1,
                                img_file=gen_img_file,
                            )

                    # increment the alpha ticker and the step
                    ticker += 1
                    step += 1

                elapsed = timeit.default_timer() - start
                elapsed = str(datetime.timedelta(seconds=elapsed)).split('.')[0]
                logger.info("Time taken for epoch: %s\n" % elapsed)

                if epoch % checkpoint_factor == 0 or epoch == 1 or epoch == epochs[current_depth]:
                    save_dir = os.path.join(output, 'models')
                    os.makedirs(save_dir, exist_ok=True)
                    gen_save_file = os.path.join(save_dir, "GAN_GEN_" + str(current_depth) + "_" + str(epoch) + ".pth")
                    dis_save_file = os.path.join(save_dir, "GAN_DIS_" + str(current_depth) + "_" + str(epoch) + ".pth")
                    gen_optim_save_file = os.path.join(
                        save_dir, "GAN_GEN_OPTIM_" + str(current_depth) + "_" + str(epoch) + ".pth")
                    dis_optim_save_file = os.path.join(
                        save_dir, "GAN_DIS_OPTIM_" + str(current_depth) + "_" + str(epoch) + ".pth")

                    torch.save(self.gen.state_dict(), gen_save_file)
                    logger.info("Saving the model to: %s\n" % gen_save_file)
                    torch.save(self.dis.state_dict(), dis_save_file)
                    torch.save(self.gen_optim.state_dict(), gen_optim_save_file)
                    torch.save(self.dis_optim.state_dict(), dis_optim_save_file)

                    # also save the shadow generator if use_ema is True
                    if self.use_ema:
                        gen_shadow_save_file = os.path.join(
                            save_dir, "GAN_GEN_SHADOW_" + str(current_depth) + "_" + str(epoch) + ".pth")
                        torch.save(self.gen_shadow.state_dict(), gen_shadow_save_file)
                        logger.info("Saving the model to: %s\n" % gen_shadow_save_file)

        logger.info('Training completed ...\n')


if __name__ == '__main__':
    print('Done.')
