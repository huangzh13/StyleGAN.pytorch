"""
Behavioral smoke tests for the synthesis/discriminator blocks in models/Blocks.py.
"""

import torch

from models.Blocks import (
    DiscriminatorBlock,
    DiscriminatorTop,
    GSynthesisBlock,
    InputBlock,
)


def test_input_block_const_output_shape():
    block = InputBlock(nf=32, dlatent_size=16, const_input_layer=True, gain=2 ** 0.5,
                       use_wscale=True, use_noise=False, use_pixel_norm=False,
                       use_instance_norm=False, use_styles=True, activation_layer=torch.nn.LeakyReLU(0.2))
    dlatents = torch.randn(2, 2, 16)  # a slice required by this block's forward
    out = block(dlatents)
    assert out.shape == (2, 32, 4, 4)
    assert torch.isfinite(out).all()


class TestGSynthesisBlock:
    def test_upsample_doubles_resolution(self):
        block = GSynthesisBlock(16, 32, blur_filter=[1, 2, 1], dlatent_size=8,
                                gain=2 ** 0.5, use_wscale=True, use_noise=False,
                                use_pixel_norm=False, use_instance_norm=True,
                                use_styles=True, activation_layer=torch.nn.LeakyReLU(0.2))
        x = torch.randn(1, 16, 8, 8)
        dlatents = torch.randn(1, 2, 8)
        out = block(x, dlatents)
        assert out.shape == (1, 32, 16, 16)

    def test_output_finite_with_noise(self):
        block = GSynthesisBlock(16, 16, blur_filter=None, dlatent_size=8,
                                gain=2 ** 0.5, use_wscale=True, use_noise=True,
                                use_pixel_norm=False, use_instance_norm=True,
                                use_styles=True, activation_layer=torch.nn.LeakyReLU(0.2))
        x = torch.randn(1, 16, 8, 8)
        dlatents = torch.randn(1, 2, 8)
        assert torch.isfinite(block(x, dlatents)).all()


class TestDiscriminatorTop:
    def test_output_shape(self):
        top = DiscriminatorTop(mbstd_group_size=4, mbstd_num_features=1,
                               in_channels=32, intermediate_channels=16,
                               gain=2 ** 0.5, use_wscale=True,
                               activation_layer=torch.nn.LeakyReLU(0.2),
                               resolution=4, output_features=1)
        out = top(torch.randn(4, 32, 4, 4))
        assert out.shape == (4, 1)


class TestDiscriminatorBlock:
    def test_half_resolution(self):
        block = DiscriminatorBlock(in_channels=32, out_channels=64, gain=2 ** 0.5,
                                   use_wscale=True, blur_kernel=[1, 2, 1],
                                   activation_layer=torch.nn.LeakyReLU(0.2))
        out = block(torch.randn(2, 32, 16, 16))
        assert out.shape == (2, 64, 8, 8)