import pytest
import torch

from src.config import Config
from src.model import MultiheadAttention, Transformer


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
    attention = MultiheadAttention(config)
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


def test_attention_accepts_shorter_runtime_sequence():
    config = Config()
    config.batch_size = 1
    config.seqlen = 32
    config.hidden_dim = 16
    config.head_n = 8
    config.device = "cpu"

    attention = MultiheadAttention(config)
    runtime_seqlen = 17
    input = torch.randn(
        config.batch_size,
        runtime_seqlen,
        config.hidden_dim,
        dtype=torch.float32,
    )

    assert attention(input).shape == (
        config.batch_size,
        runtime_seqlen,
        config.hidden_dim,
    )


class DummyTokenizer:
    def __init__(self):
        self.char_map = {"a": 0, "b": 1}
        self.char_map_rev = {0: "a", 1: "b"}
        self.eos_id = None

    def text2token(self, text: str):
        return torch.tensor([self.char_map[char] for char in text], dtype=torch.long)

    def token2text(self, tokens: torch.Tensor):
        return [self.char_map_rev[int(token.item())] for token in tokens]


def test_generate_does_not_mutate_cfg(monkeypatch):
    config = Config()
    config.batch_size = 4
    config.seqlen = 32
    config.hidden_dim = 16
    config.head_n = 8
    config.block_n = 1
    config.device = "cpu"
    config.embed = "RoPE"

    model = Transformer(config, vocab_size=2)
    tokenizer = DummyTokenizer()

    monkeypatch.setattr(torch, "multinomial", lambda probs, num_samples: torch.tensor([0]))

    original_batch_size = model.cfg.batch_size
    original_seqlen = model.cfg.seqlen

    output = model.generate("ab", tokenizer, max_new_token=1)

    assert output == "aba"
    assert model.cfg.batch_size == original_batch_size
    assert model.cfg.seqlen == original_seqlen
