"""
Tests for the yacs-based configuration parsing used by train.py and the
generation scripts.
"""

import config


def test_defaults_exist():
    assert hasattr(config.cfg, "output_dir")
    assert hasattr(config.cfg.dataset, "resolution")
    assert hasattr(config.cfg.model.gen, "latent_size")
    assert config.cfg.dataset.channels in (1, 3)


def test_merge_sample_yaml():
    cfg = config.cfg.clone()
    cfg.merge_from_file("configs/sample.yaml")
    # sample.yaml overrides resolution and structure.
    assert cfg.dataset.resolution >= 4
    assert cfg.structure in ("linear", "fixed")


def test_all_sample_configs_parse():
    import glob

    for path in glob.glob("configs/*.yaml"):
        cfg = config.cfg.clone()
        cfg.merge_from_file(path)
        assert cfg.dataset.resolution >= 4
        assert cfg.dataset.channels in (1, 3)