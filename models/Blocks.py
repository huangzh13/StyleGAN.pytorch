"""
-------------------------------------------------
   File Name:    Blocks.py
   Author:       Zhonghao Huang
   Date:         2019/10/17
   Description:
-------------------------------------------------
"""

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from models.CustomLayers import PixelNormLayer, EqualizedLinear, LayerEpilogue, EqualizedConv2d, BlurLayer, View, \
    StddevLayer


class InputBlock(nn.Module):
    """
    The first block (4x4 "pixels") doesn't have an input.
    The result of the first convolution is just replaced by a (trained) constant.
    We call it the InputBlock, the others GSynthesisBlock.
    (It might be nicer to do this the other way round,
    i.e. have the LayerEpilogue be the Layer and call the conv from that.)
    """

    def __init__(self, nf, dlatent_size, const_input_layer, gain,
                 use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer):
        super().__init__()
        self.const_input_layer = const_input_layer
        self.nf = nf

        if self.const_input_layer:
            # called 'const' in tf
            self.const = nn.Parameter(torch.ones(1, nf, 4, 4))
            self.bias = nn.Parameter(torch.ones(nf))
        else:
            self.dense = EqualizedLinear(dlatent_size, nf * 16, gain=gain / 4,
                                         use_wscale=use_wscale)
            # tweak gain to match the official implementation of Progressing GAN

        self.epi1 = LayerEpilogue(nf, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                                  use_styles, activation_layer)
        self.conv = EqualizedConv2d(nf, nf, 3, gain=gain, use_wscale=use_wscale)
        self.epi2 = LayerEpilogue(nf, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                                  use_styles, activation_layer)

    def forward(self, dlatents_in_range):
        batch_size = dlatents_in_range.size(0)

        if self.const_input_layer:
            x = self.const.expand(batch_size, -1, -1, -1)
            x = x + self.bias.view(1, -1, 1, 1)
        else:
            x = self.dense(dlatents_in_range[:, 0]).view(batch_size, self.nf, 4, 4)

        x = self.epi1(x, dlatents_in_range[:, 0])
        x = self.conv(x)
        x = self.epi2(x, dlatents_in_range[:, 1])

        return x


class GSynthesisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, blur_filter, dlatent_size, gain,
                 use_wscale, use_noise, use_pixel_norm, use_instance_norm, use_styles, activation_layer):
        # 2**res x 2**res
        # res = 3..resolution_log2
        super().__init__()

        if blur_filter:
            blur = BlurLayer(blur_filter)
        else:
            blur = None

        self.conv0_up = EqualizedConv2d(in_channels, out_channels, kernel_size=3, gain=gain, use_wscale=use_wscale,
                                        intermediate=blur, upscale=True)
        self.epi1 = LayerEpilogue(out_channels, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                                  use_styles, activation_layer)
        self.conv1 = EqualizedConv2d(out_channels, out_channels, kernel_size=3, gain=gain, use_wscale=use_wscale)
        self.epi2 = LayerEpilogue(out_channels, dlatent_size, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                                  use_styles, activation_layer)

    def forward(self, x, dlatents_in_range):
        x = self.conv0_up(x)
        x = self.epi1(x, dlatents_in_range[:, 0])
        x = self.conv1(x)
        x = self.epi2(x, dlatents_in_range[:, 1])
        return x


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

        if blur_filter is None:
            blur_filter = [1, 2, 1]

        resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** resolution_log2 and resolution >= 4
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

        assert depth < self.num_layers, "Requested output depth cannot be produced"

        if self.structure == 'fixed':
            # TODO
            x = self.init_block(dlatents_in[:, 0:2])
            for i, block in enumerate(self.blocks):
                x = block(x, dlatents_in[:, 2 * (i + 1):2 * (i + 2)])
            images_out = self.to_rgb[-1](x)
        elif self.structure == 'linear':
            # TODO
            images_out = dlatents_in
        else:
            raise KeyError("Unknown structure: ", self.structure)

        return images_out


class DiscriminatorTop(nn.Sequential):
    def __init__(self,
                 mbstd_group_size,
                 mbstd_num_features,
                 in_channels,
                 intermediate_channels,
                 gain, use_wscale,
                 activation_layer,
                 resolution=4,
                 in_channels2=None,
                 output_features=1,
                 last_gain=1):
        """
        :param mbstd_group_size:
        :param mbstd_num_features:
        :param in_channels:
        :param intermediate_channels:
        :param gain:
        :param use_wscale:
        :param activation_layer:
        :param resolution:
        :param in_channels2:
        :param output_features:
        :param last_gain:
        """

        layers = []
        if mbstd_group_size > 1:
            layers.append(('stddev_layer', StddevLayer(mbstd_group_size, mbstd_num_features)))

        if in_channels2 is None:
            in_channels2 = in_channels

        layers.append(('conv', EqualizedConv2d(in_channels + mbstd_num_features, in_channels2, kernel_size=3,
                                               gain=gain, use_wscale=use_wscale)))
        layers.append(('act0', activation_layer))
        layers.append(('view', View(-1)))
        layers.append(('dense0', EqualizedLinear(in_channels2 * resolution * resolution, intermediate_channels,
                                                 gain=gain, use_wscale=use_wscale)))
        layers.append(('act1', activation_layer))
        layers.append(('dense1', EqualizedLinear(intermediate_channels, output_features,
                                                 gain=last_gain, use_wscale=use_wscale)))

        super().__init__(OrderedDict(layers))


class DiscriminatorBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, gain, use_wscale, activation_layer):
        super().__init__(OrderedDict([
            ('conv0', EqualizedConv2d(in_channels, in_channels, kernel_size=3, gain=gain, use_wscale=use_wscale)),
            # out channels nf(res-1)
            ('act0', activation_layer),
            ('blur', BlurLayer()),
            ('conv1_down', EqualizedConv2d(in_channels, out_channels, kernel_size=3,
                                           gain=gain, use_wscale=use_wscale, downscale=True)),
            ('act1', activation_layer)]))


if __name__ == '__main__':
    g_mapping = GMapping()
    g_synthesis = GSynthesis()
    # discriminator = DiscriminatorTop()
    print('Done.')
