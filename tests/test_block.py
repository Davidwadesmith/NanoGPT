import torch
import pytest
from src.model import Block
from src.config import Config


@pytest.mark.parametrize(
    "batch_size, seqlen, hidden_dim, head_n, device", [(1, 16, 16, 8, "cpu")]
)
def test_attention_shape(batch_size, seqlen, hidden_dim, head_n, device):
    config = Config()
    config.batch_size = batch_size
    config.seqlen = seqlen
    config.hidden_dim = hidden_dim
    config.head_n = head_n
    config.device = device
    attention = Block(config)
    gen = torch.Generator(device)
    input = torch.randn(
        config.batch_size,
        config.seqlen,
        config.hidden_dim,
        generator=gen,
        dtype=torch.float32,
    ).to(torch.device(device))
    assert attention(input).shape == (
        config.batch_size,
        config.seqlen,
        config.hidden_dim,
    )
