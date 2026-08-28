import numpy as np
import pytest
import torch

from generate_samples import load_dlatent


def test_load_dlatent_accepts_saved_w_code(tmp_path):
    path = tmp_path / "w.npy"
    np.save(path, np.zeros((6, 4), dtype=np.float64))

    code = load_dlatent(path, num_layers=6, dlatent_size=4)

    assert code.shape == (1, 6, 4)
    assert code.dtype == torch.float32


def test_load_dlatent_rejects_wrong_shape(tmp_path):
    path = tmp_path / "w.npy"
    np.save(path, np.zeros((4,), dtype=np.float32))

    with pytest.raises(ValueError, match="Expected a W/dlatent code"):
        load_dlatent(path, num_layers=6, dlatent_size=4)
