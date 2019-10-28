"""
-------------------------------------------------
   File Name:    Blocks.py
   Date:         2019/10/17
   Description:  Copy from: https://github.com/lernapparat/lernapparat
-------------------------------------------------
"""

from collections import OrderedDict

import torch
import torch.nn as nn

from models.CustomLayers import EqualizedLinear, LayerEpilogue, EqualizedConv2d, BlurLayer, View, StddevLayer


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
    def __init__(self, in_channels, out_channels, gain, use_wscale, activation_layer, blur_kernel):
        super().__init__(OrderedDict([
            ('conv0', EqualizedConv2d(in_channels, in_channels, kernel_size=3, gain=gain, use_wscale=use_wscale)),
            # out channels nf(res-1)
            ('act0', activation_layer),
            ('blur', BlurLayer(kernel=blur_kernel)),
            ('conv1_down', EqualizedConv2d(in_channels, out_channels, kernel_size=3,
                                           gain=gain, use_wscale=use_wscale, downscale=True)),
            ('act1', activation_layer)]))


if __name__ == '__main__':
    # discriminator = DiscriminatorTop()
    print('Done.')
