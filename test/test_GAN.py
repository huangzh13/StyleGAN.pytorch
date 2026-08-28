"""
Smoke tests for the StyleGAN network containers (gensynthesis/gan.py):
mapping network, generator, and discriminator construction and forward passes.

Run on CPU with small resolutions and latent sizes so the suite stays fast.
These verify internal consistency only, not NVIDIA numerical equivalence.
"""

import torch

from models.GAN import Discriminator, GMapping, Generator


def _full_depth(gen: Generator) -> int:
    """Depth that yields the full-resolution image (max valid depth)."""
    return gen.g_synthesis.depth - 1


class TestGMapping:
    def test_broadcast_shape(self):
        net = GMapping(latent_size=8, dlatent_size=8, dlatent_broadcast=4,
                       mapping_layers=2, mapping_fmaps=8)
        out = net(torch.randn(3, 8))
        assert out.shape == (3, 4, 8)

    def test_forward_finite(self):
        net = GMapping(latent_size=8, dlatent_size=8, mapping_layers=2, mapping_fmaps=8)
        assert torch.isfinite(net(torch.randn(2, 8))).all()


class TestGenerator:
    def test_constructs(self):
        gen = Generator(resolution=16, latent_size=4, fmap_base=128,
                        truncation_psi=0.0, style_mixing_prob=0.0)
        assert sum(p.numel() for p in gen.parameters()) > 0

    def test_full_depth_forward_shape(self):
        gen = Generator(resolution=16, latent_size=4, fmap_base=128,
                        truncation_psi=0.0, style_mixing_prob=0.0).eval()
        depth = _full_depth(gen)
        img = gen(torch.randn(2, 4), depth, 0.0)
        assert img.shape == (2, 3, 16, 16)
        assert torch.isfinite(img).all()

    def test_fade_in_alpha_path(self):
        gen = Generator(resolution=16, latent_size=4, fmap_base=128,
                        truncation_psi=0.0, style_mixing_prob=0.0).eval()
        depth = _full_depth(gen)
        img = gen(torch.randn(2, 4), depth, 0.5)
        assert img.shape == (2, 3, 16, 16)

    def test_w_code_can_drive_synthesis(self):
        gen = Generator(resolution=16, latent_size=4, dlatent_size=6,
                        fmap_base=128, mapping_layers=2,
                        truncation_psi=0.0, style_mixing_prob=0.0).eval()
        depth = _full_depth(gen)
        w = gen.g_mapping(torch.randn(1, 4))
        assert w.shape == (1, gen.num_layers, 6)
        img = gen.g_synthesis(w, depth=depth, alpha=1.0)
        assert img.shape == (1, 3, 16, 16)
        assert torch.isfinite(img).all()


class TestDiscriminator:
    def test_constructs_and_forward(self):
        dis = Discriminator(resolution=16, num_channels=3, mbstd_group_size=4).eval()
        depth = dis.depth - 1
        out = dis(torch.randn(2, 3, 16, 16), depth, 1.0)
        assert out.shape == (2, 1)
        assert torch.isfinite(out).all()
