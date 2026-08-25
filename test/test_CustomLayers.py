"""
Behavioral smoke tests for the custom layers in models/CustomLayers.py.

These tests verify high-value invariants (shapes, finite values, and small
numerical properties) on CPU with deterministic inputs. They are not a claim of
numerical equivalence with NVIDIA's implementation.
"""

import torch

from models.CustomLayers import (
    BlurLayer,
    EqualizedConv2d,
    EqualizedLinear,
    NoiseLayer,
    PixelNormLayer,
    StddevLayer,
    Truncation,
    Upscale2d,
)


class TestPixelNorm:
    def test_output_shape_matches_input(self):
        layer = PixelNormLayer()
        x = torch.randn(4, 8, 16, 16)
        assert layer(x).shape == x.shape

    def test_normalizes_per_pixel(self):
        layer = PixelNormLayer(epsilon=1e-8)
        x = torch.randn(4, 8, 16, 16)
        out = layer(x)
        # Reference: divide by the per-channel RMS (matches the layer formula).
        expected = x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_zero_input_stays_finite(self):
        layer = PixelNormLayer()
        x = torch.zeros(2, 3, 4, 4)
        assert torch.isfinite(layer(x)).all()


class TestEqualizedLinear:
    def test_output_shape(self):
        layer = EqualizedLinear(8, 16, use_wscale=True)
        out = layer(torch.randn(3, 8))
        assert out.shape == (3, 16)

    def test_no_bias_variant(self):
        layer = EqualizedLinear(4, 5, bias=False)
        assert layer.bias is None
        assert layer(torch.randn(2, 4)).shape == (2, 5)

    def test_output_finite(self):
        layer = EqualizedLinear(8, 8, use_wscale=True)
        assert torch.isfinite(layer(torch.randn(4, 8))).all()


class TestEqualizedConv2d:
    def test_output_shape_same_resolution(self):
        layer = EqualizedConv2d(3, 16, 3, use_wscale=True)
        out = layer(torch.randn(2, 3, 8, 8))
        assert out.shape == (2, 16, 8, 8)

    def test_upscale_doubles_resolution(self):
        layer = EqualizedConv2d(2, 4, 3, upscale=True, use_wscale=True)
        out = layer(torch.randn(1, 2, 4, 4))
        assert out.shape == (1, 4, 8, 8)

    def test_output_finite(self):
        layer = EqualizedConv2d(3, 3, 3, use_wscale=True)
        assert torch.isfinite(layer(torch.randn(1, 3, 4, 4))).all()


class TestStddevLayer:
    def test_adds_features(self):
        x = torch.randn(8, 3, 8, 8)
        out = StddevLayer(group_size=4, num_new_features=1)(x)
        assert out.shape == (8, 4, 8, 8)

    def test_group_size_larger_than_batch_clamps(self):
        x = torch.randn(3, 2, 8, 8)  # batch < requested group size
        out = StddevLayer(group_size=4, num_new_features=1)(x)
        assert out.shape == (3, 3, 8, 8)


class TestBlurLayer:
    def test_interior_of_constant_tensor_preserved(self):
        # Zero-padding at the border shrinks edge values; the interior of a
        # constant input must be reproduced (normalized kernel sums to 1).
        layer = BlurLayer(kernel=[1, 2, 1], normalize=True)
        x = torch.full((1, 1, 16, 16), 2.0)
        out = layer(x)
        assert torch.allclose(out[:, :, 1:-1, 1:-1], x[:, :, 1:-1, 1:-1], atol=1e-6)

    def test_preserves_shape(self):
        layer = BlurLayer(kernel=[1, 2, 1])
        x = torch.randn(2, 3, 6, 6)
        assert layer(x).shape == x.shape


class TestUpscale2d:
    def test_doubles_resolution(self):
        layer = Upscale2d()
        out = layer(torch.randn(2, 4, 4, 4))
        assert out.shape == (2, 4, 8, 8)


class TestNoiseLayer:
    def test_injected_noise_makes_deterministic_output(self):
        layer = NoiseLayer(channels=3)
        layer.weight.data.fill_(2.0)
        x = torch.ones(1, 3, 4, 4)
        noise = torch.ones(1, 1, 4, 4)
        out = layer(x, noise=noise)
        expected = x + 2.0 * noise
        assert torch.allclose(out, expected)


class TestTruncation:
    def test_forward_applies_lerp_for_first_layers(self):
        avg = torch.zeros(3)
        t = Truncation(avg_latent=avg, max_layer=2, threshold=0.7)
        x = torch.randn(2, 5, 3)
        out = t(x)
        assert out.shape == x.shape
        # First max_layer values are lerped toward avg; the rest are untouched.
        assert torch.allclose(out[:, :2], 0.7 * x[:, :2])
        assert torch.allclose(out[:, 2:], x[:, 2:])

    def test_update_moves_average(self):
        t = Truncation(avg_latent=torch.zeros(3), beta=0.9)
        t.update(torch.full((3,), 10.0))
        assert torch.allclose(t.avg_latent, torch.full((3,), 1.0))