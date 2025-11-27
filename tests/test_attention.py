from torch import randint
import pytest
from src.model import MultiheadAttention
from src.config import Config


@pytest.mark.parametrize(
    "batch_size, seqlen, hidden_dim, head_n, device", [(1, 16, 16, 8, "cuda")]
)
def test_attention_shape(batch_size, seqlen, hidden_dim, head_n, device):
    config = Config()
    config.batch_size = batch_size
    config.seqlen = seqlen
    config.hidden_dim = hidden_dim
    config.head_n = head_n
    config.device = device
    attention = MultiheadAttention(config)
    input = randint(0, 1000, (config.batch_size, config.seqlen, config.hidden_dim))
    assert attention(input).shape == (
        config.batch_size,
        config.seqlen,
        config.hidden_dim,
    )
