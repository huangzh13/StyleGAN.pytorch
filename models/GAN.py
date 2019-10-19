"""
-------------------------------------------------
   File Name:    GAN.py
   Author:       Zhonghao Huang
   Date:         2019/10/17
   Description:
-------------------------------------------------
"""

import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import AvgPool2d

from models.Blocks import GMapping, GSynthesis, DiscriminatorTop, DiscriminatorBlock
from models.CustomLayers import EqualizedConv2d


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
    def __init__(self,
                 num_channels=1,
                 resolution=32,
                 # label_size=0,  # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
                 fmap_base=8192,
                 fmap_decay=1.0,
                 fmap_max=512,
                 nonlinearity='lrelu',
                 use_wscale=True,
                 mbstd_group_size=4,
                 mbstd_num_features=1,
                 blur_filter=None,
                 structure='liner',
                 **kwargs):
        """
        Discriminator used in the StyleGAN paper.
        :param num_channels: Number of input color channels. Overridden based on dataset.
        :param resolution: Input resolution. Overridden based on dataset.
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
        final_block = [('4x4', DiscriminatorTop(**kwargs))]
        self.top = nn.Sequential(OrderedDict(final_block))

        # create the fromRGB layers for various inputs:
        from_rgb = [EqualizedConv2d(**kwargs)]

        # create the remaining layers
        blocks = []
        for res in range(3, resolution_log2 + 1):
            last_channels = nf(res - 2)
            channels = nf(res - 1)
            name = '{s}x{s}'.format(s=2 ** res)
            blocks.append((name, DiscriminatorBlock(**kwargs)))
            # create the fromRGB layers for various inputs:
            from_rgb.append(EqualizedConv2d(**kwargs))

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

    def __init__(self, depth, latent_size
                 ):
        pass

        # Create the Generator and the Discriminator
        # if code is to be run on GPU, we can use DataParallel:
        # state of the object
        # define the optimizers for the discriminator and generator
        # define the loss function used for training the GAN
        # Use of ema

    def __setup_loss(self, loss):
        pass

    def __progressive_down_sampling(self, real_batch, depth, alpha):
        pass

    def optimize_discriminator(self, noise, real_batch, depth, alpha):
        """
        performs one step of weight update on discriminator using the batch of data
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
        :param depth: current depth of optimization
        :param alpha: current alpha for fade-in
        :return: current loss (Wasserstein loss)
        """
        pass

    def optimize_generator(self, noise, real_batch, depth, alpha):
        """
        performs one step of weight update on generator for the given batch_size
        :param noise: input random noise required for generating samples
        :param real_batch: batch of real samples
        :param depth: depth of the network at which optimization is done
        :param alpha: value of alpha for fade-in effect
        :return: current loss (Wasserstein estimate)
        """

    def train(self, dataset, epochs, batch_sizes,
              fade_in_percentage, num_samples=16,
              start_depth=0, num_workers=3, feedback_factor=100,
              log_dir="./models/", sample_dir="./samples/", save_dir="./models/",
              checkpoint_factor=1):
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


if __name__ == '__main__':
    print('Done.')
