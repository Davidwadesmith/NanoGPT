import pytest
import torch

from src.model import MultiheadAttention


@pytest.mark.parametrize("batch_size, seqlen, hidden_dim, device", [(2, 16, 32, "cpu")])
def test_RoPE_shape(batch_size, seqlen, hidden_dim, device):
    x = torch.randn((batch_size, seqlen, hidden_dim))
    result = MultiheadAttention.RoPE(
        batch_size, seqlen, hidden_dim, x, torch.device(device)
    )
    assert result.shape == x.shape
