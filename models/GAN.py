"""
-------------------------------------------------
   File Name:    GAN.py
   Author:       Zhonghao Huang
   Date:         2019/10/17
   Description:
-------------------------------------------------
"""

import numpy as np

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self,
                 mapping_fmaps=512,
                 style_mixing_prob=0.9,  # Probability of mixing styles during training. None = disable.
                 truncation_psi=0.7,  # Style strength multiplier for the truncation trick. None = disable.
                 truncation_cutoff=8,  # Number of layers for which to apply the truncation trick. None = disable.
                 **kwargs
                 ):
        super(Generator, self).__init__()

        self.mapping_fmaps = mapping_fmaps
        self.style_mixing_prob = style_mixing_prob
        self.truncation_cutoff = truncation_cutoff
        self.truncation_psi = truncation_psi

        self.g_mapping = G_mapping(self.mapping_fmaps, **kwargs)
        self.g_synthesis = G_synthesis(self.mapping_fmaps, **kwargs)

    def forward(self, x):
        dlatents_in = self.g_mapping(x)
        fake_imgs = self.g_synthesis(dlatents_in)

        return fake_imgs


class Discriminator(nn.Module):
    def __init__(self,
                 # images_in,     # First input: Images [minibatch, channel, height, width].
                 # labels_in,     # Second input: Labels [minibatch, label_size].
                 num_channels=3,  # Number of input color channels. Overridden based on dataset.
                 resolution=1024,  # Input resolution. Overridden based on dataset.
                 fmap_base=8192,  # Overall multiplier for the number of feature maps.
                 fmap_decay=1.0,  # log2 feature map reduction when doubling the resolution.
                 fmap_max=512,  # Maximum number of feature maps in any layer.
                 nonlinearity='lrelu',  # Activation function: 'relu', 'lrelu',
                 use_wscale=True,  # Enable equalized learning rate?
                 mbstd_group_size=4,  # Group size for the minibatch standard deviation layer, 0 = disable.
                 mbstd_num_features=1,  # Number of features for the minibatch standard deviation layer.
                 # blur_filter = [1,2,1], # Low-pass filter to apply when resampling activations. None = no filtering.
                 ):
        self.mbstd_group_size = mbstd_group_size
        self.mbstd_num_features = mbstd_num_features
        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4

        def nf(stage):
            return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        act, gain = {'relu': (torch.relu, np.sqrt(2)),
                     'lrelu': (nn.LeakyReLU(negative_slope=0.2), np.sqrt(2))}[nonlinearity]

        self.gain = gain
        self.use_wscale = use_wscale

    def forward(self, x):
        pass


class StyleGAN:
    """ Wrapper around the Generator and the Discriminator """

    def __init__(self, depth, latent_size
                 ):
        """
        constructor for the class
        :param depth: depth of the GAN (will be used for each generator and discriminator)
        :param latent_size: latent size of the manifold used by the GAN
        :param learning_rate: learning rate for Adam
        :param beta_1: beta_1 for Adam
        :param beta_2: beta_2 for Adam
        :param eps: epsilon for Adam
        :param n_critic: number of times to update discriminator per generator update
        :param drift: drift penalty for the
                    (Used only if loss is wgan or wgan-gp)
        :param use_eql: whether to use equalized learning rate
        :param loss: the loss function to be used
                             Can either be a string =>
                                  ["wgan-gp", "wgan", "lsgan", "lsgan-with-sigmoid",
                                  "hinge", "standard-gan" or "relativistic-hinge"]
                             Or an instance of GANLoss
        :param use_ema: boolean for whether to use exponential moving averages
        :param ema_decay: value of mu for ema
        :param device: device to run the GAN on (GPU / CPU)
        """
        # Create the Generator and the Discriminator

        # if code is to be run on GPU, we can use DataParallel:

        # state of the object
        self.latent_size = latent_size
        self.depth = depth
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.n_critic = n_critic
        self.use_eql = use_eql
        self.device = device
        self.drift = drift

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
