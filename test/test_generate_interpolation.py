import torch

from generate_interpolation import slerp


def test_slerp_preserves_endpoints():
    start = torch.tensor([[1.0, 0.0]])
    end = torch.tensor([[0.0, 1.0]])

    assert torch.allclose(slerp(start, end, 0.0), start)
    assert torch.allclose(slerp(start, end, 1.0), end)
