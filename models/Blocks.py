"""
-------------------------------------------------
   File Name:    Blocks.py
   Author:       Zhonghao Huang
   Date:         2019/10/17
   Description:
-------------------------------------------------
"""

import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn

from models.CustomLayers import PixelNormLayer, EqualizedLinear


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
        x = self.map(x)
        # Broadcast -> batch_size * dlatent_broadcast * dlatent_size
        if self.dlatent_broadcast is not None:
            x = x.unsqueeze(1).expand(-1, self.dlatent_broadcast, -1)
        return x


class GSynthesis(nn.Module):
    """
    
    """

    def forward(self, x):
        pass


if __name__ == '__main__':
    g_mapping = GMapping()
    print('Done.')
