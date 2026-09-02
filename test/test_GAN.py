"""
Smoke tests for the StyleGAN network containers (gensynthesis/gan.py):
mapping network, generator, and discriminator construction and forward passes.

Run on CPU with small resolutions and latent sizes so the suite stays fast.
These verify internal consistency only, not NVIDIA numerical equivalence.
"""

import torch

from models.GAN import (GENERATOR_GRADIENT_CLIP_NORM, Discriminator, GMapping,
                        Generator, StyleGAN)
from models.CustomLayers import NoiseLayer


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

    def test_fixed_noise_makes_synthesis_deterministic(self):
        gen = Generator(resolution=8, latent_size=4, dlatent_size=6,
                        fmap_base=128, mapping_layers=2,
                        truncation_psi=0.0, style_mixing_prob=0.0).eval()
        for module in gen.g_synthesis.modules():
            if isinstance(module, NoiseLayer):
                module.weight.data.fill_(1.0)

        gen.set_noise_seed(23)
        w = gen.g_mapping(torch.randn(1, 4))
        first = gen.g_synthesis(w, depth=1, alpha=1.0)
        second = gen.g_synthesis(w, depth=1, alpha=1.0)

        assert torch.equal(first, second)

    def test_conditional_forward_uses_labels(self):
        gen = Generator(resolution=8, latent_size=4, dlatent_size=8,
                        mapping_layers=2, conditional=True, n_classes=2,
                        fmap_base=128, truncation_psi=0.0,
                        style_mixing_prob=0.0).eval()
        labels = torch.tensor([0, 1], dtype=torch.long)
        img = gen(torch.randn(2, 4), depth=1, alpha=1.0, labels_in=labels)

        # The progressive discriminator assumes the low-resolution feature
        # maps are capped at the same width as the historical default. Keep
        # this smoke test aligned with that supported configuration.
        dis = Discriminator(resolution=8, conditional=True, n_classes=2,
                            fmap_base=2048, mbstd_group_size=2).eval()
        scores = dis(img, depth=1, alpha=1.0, labels_in=labels)

        assert img.shape == (2, 3, 8, 8)
        assert scores.shape == (2, 1)
        assert torch.isfinite(scores).all()


class TestDiscriminator:
    def test_constructs_and_forward(self):
        dis = Discriminator(resolution=16, num_channels=3, mbstd_group_size=4).eval()
        depth = dis.depth - 1
        out = dis(torch.randn(2, 3, 16, 16), depth, 1.0)
        assert out.shape == (2, 1)
        assert torch.isfinite(out).all()


def test_conditional_training_step_passes_dataset_labels():
    optimizer_args = {
        "learning_rate": 0.001,
        "beta_1": 0.0,
        "beta_2": 0.99,
        "eps": 1e-8,
    }
    gan = StyleGAN(
        structure="linear",
        resolution=4,
        num_channels=3,
        latent_size=4,
        g_args={
            "latent_size": 4,
            "dlatent_size": 8,
            "mapping_layers": 2,
            "fmap_base": 128,
            "truncation_psi": 0.0,
            "style_mixing_prob": 0.0,
        },
        d_args={"fmap_base": 128, "mbstd_group_size": 2},
        g_opt_args=optimizer_args,
        d_opt_args=optimizer_args,
        conditional=True,
        n_classes=2,
        loss="conditional-loss",
        use_ema=False,
        device=torch.device("cpu"),
    )
    noise = torch.randn(2, 4)
    real = torch.randn(2, 3, 4, 4)
    labels = torch.tensor([0, 1], dtype=torch.long)

    dis_loss = gan.optimize_discriminator(noise, real, depth=0, alpha=1.0,
                                          labels=labels)
    gen_loss = gan.optimize_generator(noise, real, depth=0, alpha=1.0,
                                      labels=labels)

    assert torch.isfinite(torch.tensor(dis_loss))
    assert torch.isfinite(torch.tensor(gen_loss))


def test_generator_training_step_clips_gradient_norm(monkeypatch):
    """The generator optimizer step must sanitize an oversized gradient."""
    optimizer_args = {
        "learning_rate": 0.001,
        "beta_1": 0.0,
        "beta_2": 0.99,
        "eps": 1e-8,
    }
    gan = StyleGAN(
        structure="linear",
        resolution=4,
        num_channels=3,
        latent_size=4,
        g_args={
            "latent_size": 4,
            "dlatent_size": 8,
            "mapping_layers": 2,
            "fmap_base": 128,
            "truncation_psi": 0.0,
            "style_mixing_prob": 0.0,
        },
        d_args={"fmap_base": 128, "mbstd_group_size": 2},
        g_opt_args=optimizer_args,
        d_opt_args=optimizer_args,
        loss="logistic",
        use_ema=False,
        device=torch.device("cpu"),
    )

    # Force the step to exceed the clipping threshold so this verifies
    # clipping itself rather than merely the presence of the call.
    parameter = next(param for param in gan.gen.parameters()
                     if param.requires_grad)
    hook = parameter.register_hook(lambda gradient: gradient * 1e6)
    observed = {}
    original_clip = torch.nn.utils.clip_grad_norm_

    def recording_clip(parameters, max_norm, *args, **kwargs):
        parameters = list(parameters)
        observed["before"] = torch.linalg.vector_norm(torch.stack([
            param.grad.detach().norm(2).to(torch.float64)
            for param in parameters if param.grad is not None
        ])).item()
        observed["max_norm"] = max_norm
        original_clip(parameters, max_norm, *args, **kwargs)
        observed["after"] = torch.linalg.vector_norm(torch.stack([
            param.grad.detach().norm(2).to(torch.float64)
            for param in parameters if param.grad is not None
        ])).item()

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", recording_clip)

    try:
        gan.optimize_generator(
            torch.randn(2, 4),
            torch.randn(2, 3, 4, 4),
            depth=0,
            alpha=1.0,
        )
    finally:
        hook.remove()

    assert observed["max_norm"] == GENERATOR_GRADIENT_CLIP_NORM
    assert observed["before"] > GENERATOR_GRADIENT_CLIP_NORM
    assert observed["after"] <= GENERATOR_GRADIENT_CLIP_NORM * (1 + 1e-6)
