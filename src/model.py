"""
NanoGPT的类定义代码
"""

from math import inf
from typing import OrderedDict, Tuple
import logging
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn

from src.tokenizer import Tokenizer
from .config import Config
from .dataloader import DataLoader

# ---Debug Settings
logging.basicConfig(
    level=logging.ERROR,
    format=logging.BASIC_FORMAT,
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


# ---global constant---
torch.manual_seed(114514)


class MultiheadAttention(torch.nn.Module):
    def __init__(self, cfg: Config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        hidden_dim = cfg.hidden_dim
        self.wq = nn.Linear(hidden_dim, hidden_dim, bias=False, device=cfg.device)
        self.wk = nn.Linear(hidden_dim, hidden_dim, bias=False, device=cfg.device)
        self.wv = nn.Linear(hidden_dim, hidden_dim, bias=False, device=cfg.device)
        self.wo = nn.Linear(hidden_dim, hidden_dim, bias=False, device=cfg.device)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden_tokens) -> torch.Tensor:
        q = self.wq(hidden_tokens)
        k = self.wk(hidden_tokens)
        if self.cfg.embed == "RoPE":
            q = self.RoPE(
                self.cfg.batch_size,
                self.cfg.seqlen,
                self.cfg.hidden_dim,
                q,
                torch.device(self.cfg.device),
            )
            k = self.RoPE(
                self.cfg.batch_size,
                self.cfg.seqlen,
                self.cfg.hidden_dim,
                k,
                torch.device(self.cfg.device),
            )

        q = torch.transpose(
            q.view(
                self.cfg.batch_size,
                self.cfg.seqlen,
                self.cfg.head_n,
                self.cfg.hidden_dim // self.cfg.head_n,
            ),
            1,
            2,
        )  # (batch_size, head_n, seqlen, head_dim)
        k = torch.transpose(
            k.view(
                self.cfg.batch_size,
                self.cfg.seqlen,
                self.cfg.head_n,
                self.cfg.hidden_dim // self.cfg.head_n,
            ),
            1,
            2,
        )  # (batch_size, head_n, seqlen, head_dim)
        v = torch.transpose(
            self.wv(hidden_tokens).view(
                self.cfg.batch_size,
                self.cfg.seqlen,
                self.cfg.head_n,
                self.cfg.hidden_dim // self.cfg.head_n,
            ),
            1,
            2,
        )  # (batch_size, head_n, seqlen, head_dim)

        attention_map = (
            q  # (batch_size, head_n, seqlen, head_dim)
            @ torch.transpose(k, -2, -1)  # (batch_size, head_n, head_dim, seqlen)
            / ((self.cfg.hidden_dim // self.cfg.head_n) ** 0.5)
        )  # (batch_size, head_n, seqlen, seqlen)

        self.mask = (torch.tril(torch.ones(self.cfg.seqlen, self.cfg.seqlen)) == 0).to(
            torch.device(self.cfg.device)
        )  # (seqlen, seqlen)

        masked_attention_map = torch.masked_fill(
            attention_map, self.mask, -inf
        )  # match -1, -2 dim
        masked_attention = (
            self.softmax(masked_attention_map) @ v
        )  # (batch_size, head_n, seqlen, head_dim)

        return self.wo(
            torch.transpose(masked_attention, 1, 2).reshape(
                self.cfg.batch_size, self.cfg.seqlen, self.cfg.hidden_dim
            )
        )  # (batch_size, seqlen, hidden_dim)

    @staticmethod
    def RoPE(
        batch_size: int,
        seqlen: int,
        hidden_dim: int,
        x: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        result = torch.zeros_like(x).to(device)  # (batch_size, seqlen, hidden_dim)
        position = (
            torch.arange(0, seqlen, dtype=torch.float32).unsqueeze(1).unsqueeze(0)
        ).to(device)  # (1, seqlen, 1)
        pe = (
            torch.exp(
                (-torch.arange(0, hidden_dim, 2, dtype=torch.float32) + 2)
                / hidden_dim
                * torch.log(torch.tensor(10000.0))
            )
            .unsqueeze(0)
            .unsqueeze(0)
        ).to(device)  # (1, 1, d // 2)
        result[:, :, : hidden_dim // 2] = x[:, :, : hidden_dim // 2] * torch.cos(
            pe * position
        ) - x[:, :, hidden_dim // 2 :] * torch.sin(pe * position)

        result[:, :, hidden_dim // 2 :] = x[:, :, : hidden_dim // 2] * torch.sin(
            pe * position
        ) + x[:, :, hidden_dim // 2 :] * torch.cos(pe * position)

        return result


class FFN(nn.Module):
    def __init__(self, cfg: Config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        hidden_dim = cfg.hidden_dim
        self.fastforward_layer_in = nn.Linear(
            hidden_dim, hidden_dim * 8, bias=False, device=cfg.device
        )
        self.gelu = nn.GELU()
        self.fastforward_layer_out = nn.Linear(
            hidden_dim * 8, hidden_dim, bias=False, device=cfg.device
        )

    def forward(self, hidden_tokens: torch.Tensor) -> torch.Tensor:
        return self.fastforward_layer_out(
            self.gelu(self.fastforward_layer_in(hidden_tokens))
        )


class SwiGLU(nn.Module):
    def __init__(self, cfg: Config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.w1 = nn.Linear(
            cfg.hidden_dim, cfg.hidden_dim * 3, bias=False, device=cfg.device
        )
        self.w2 = nn.Linear(
            cfg.hidden_dim, cfg.hidden_dim * 3, bias=False, device=cfg.device
        )
        self.w3 = nn.Linear(
            cfg.hidden_dim * 3, cfg.hidden_dim, bias=False, device=cfg.device
        )
        self.silu = nn.SiLU()

    def forward(self, hidden_tokens) -> torch.Tensor:
        return self.w3(self.silu(self.w2(hidden_tokens)) * self.w1(hidden_tokens))


class LayerNorm(nn.Module):
    def __init__(self, cfg: Config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.gamma = nn.Parameter(torch.ones(cfg.hidden_dim), requires_grad=True).to(
            torch.device(cfg.device)
        )

        self.beta = nn.Parameter(torch.zeros(cfg.hidden_dim), requires_grad=True).to(
            torch.device(cfg.device)
        )

    def forward(self, hidden_tokens: torch.Tensor) -> torch.Tensor:
        mean = hidden_tokens.mean(dim=-1, keepdim=True)  # mean(batch_size, seqlen, 1)
        variance = torch.var(
            hidden_tokens, dim=-1, keepdim=True
        )  # variance(batch_size, seqlen, 1)
        x = (hidden_tokens - mean) / (variance**0.5)
        y = (self.gamma * x) / ((variance + 1e-5) ** 0.5) + self.beta
        return y


class RMSNorm(nn.Module):
    def __init__(self, cfg: Config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.gamma = nn.Parameter(torch.ones(cfg.hidden_dim), requires_grad=True).to(
            torch.device(cfg.device)
        )

    def forward(self, hidden_tokens: torch.Tensor) -> torch.Tensor:
        variance = torch.var(
            hidden_tokens, dim=-1, keepdim=True
        )  # variance(batch_size, seqlen, 1)
        x = (hidden_tokens) / (variance**0.5)
        y = (self.gamma * x) / ((variance + 1e-5) ** 0.5)
        return y


class Block(nn.Module):
    def __init__(self, cfg: Config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.multiheadAttention = MultiheadAttention(cfg)

        if cfg.norm == "LayerNorm":
            self.layernorm_1 = LayerNorm(cfg)
            self.layernorm_2 = LayerNorm(cfg)
        elif cfg.norm == "RMSNorm":
            self.layernorm_1 = RMSNorm(cfg)
            self.layernorm_2 = RMSNorm(cfg)

        if cfg.activation == "GELU":
            self.ffn = FFN(cfg)
        elif cfg.activation == "SwiGLU":
            self.ffn = SwiGLU(cfg)

    def forward(self, hidden_tokens: torch.Tensor) -> torch.Tensor:
        x = self.multiheadAttention(self.layernorm_1(hidden_tokens)) + hidden_tokens
        return self.ffn(self.layernorm_2(x)) + x


class Transformer(nn.Module):
    def __init__(
        self,
        cfg: Config = Config(),
        vocab_size: int = Config().vocab_size,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.blocks = nn.Sequential(
            OrderedDict([(f"{i}", Block(cfg)) for i in range(cfg.block_n)])
        )
        self.embedding_layer = nn.Embedding(
            vocab_size, cfg.hidden_dim, device=cfg.device
        )
        self.linear = nn.Linear(
            cfg.hidden_dim, vocab_size, bias=False, device=cfg.device
        )
        self.linear.weight = self.embedding_layer.weight

    def forward(
        self, tokens, target_tokens=None
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        tokens = tokens.to(torch.device(self.cfg.device))  # tokens(batch_size, seqlen)
        if target_tokens is not None:
            target_tokens = target_tokens.to(
                torch.device(self.cfg.device)
            )  # tokens(batch_size, seqlen)

        batch_size = tokens.shape[0]
        seqlen = tokens.shape[1]

        embeddings = self.embedding_layer(
            tokens
        )  # embeddings(batch_size, seqlen, hidden_dim)

        self.positional_embedding_layer = self.PositionEmbedding(
            self.cfg.seqlen, self.cfg.hidden_dim
        ).to(torch.device(self.cfg.device))

        if self.cfg.embed == "sinusoidal":
            embeddings = embeddings + self.positional_embedding_layer
            logger.debug(f"{embeddings.shape=}")
        elif self.cfg.embed == "RoPE":
            pass
        else:
            logger.info("Default config of positional embeddings: sinusoidal")
            embeddings = embeddings + self.positional_embedding_layer
            logger.debug(f"{embeddings.shape=}")

        hidden = self.blocks(embeddings)  # hidden(batch_size, seqlen, hidden_dim)
        last = self.linear(hidden)  # last(batch_size, seqlen, hidden_dim)

        new_logits = F.softmax(
            last, dim=-1
        )  # new_logits(batch_size, seqlen, hidden_dim)

        if target_tokens is not None:
            loss = F.cross_entropy(
                last.view(batch_size * seqlen, -1),  # (batch_size*seqlen, hidden_dim)
                target_tokens.view(batch_size * seqlen),  # (batch_size*seqlen)
            )
            return new_logits, loss
        else:
            return new_logits

    def generate(self, text: str, tokenizer: Tokenizer, max_new_token=50) -> str:
        logger.critical(f"{text=}")
        cfg = self.cfg
        tmp_batch_size = cfg.batch_size
        cfg.batch_size = 1
        self.cfg = cfg

        tokens = (
            tokenizer.text2token(text).unsqueeze(0).to(torch.device(self.cfg.device))
        )
        max_context = self.cfg.seqlen

        assert tokens.ndim == 2
        for _ in range(max_new_token):
            self.cfg.seqlen = min(len(tokens[0]), max_context)
            logger.critical(f"{self.cfg.seqlen=}")
            new_token = torch.multinomial(
                self(tokens[:, -self.cfg.seqlen :]).squeeze(0)[-1], num_samples=1
            )
            tokens = torch.cat([tokens, new_token.unsqueeze_(0)], dim=-1)

        cfg = self.cfg
        cfg.batch_size = tmp_batch_size
        self.cfg = cfg
        return "".join([tokenizer.token2text(t) for t in tokens][0])

    @staticmethod
    def PositionEmbedding(seqlen: int, hidden_dim: int):
        position = torch.arange(seqlen).unsqueeze(1)
        pe = torch.zeros(seqlen, hidden_dim)
        div_term = torch.exp(
            -torch.arange(0, hidden_dim, 2)
            / hidden_dim
            * torch.log(torch.tensor(10000.0))
        )
        pe[:, ::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe


class BigramModel(torch.nn.Module):
    def __init__(
        self, cfg: Config, vocab_size: int, hidden_dim: int = 120, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.embedding_layer = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size, bias=False, device=cfg.device)

    def forward(self, tokens, target_tokens=None):
        batch_size = len(tokens)
        seqlen = len(tokens[0])

        embeddings = self.embedding_layer(tokens)
        hidden = self.linear(embeddings)
        # print(embeddings.shape)
        new_logits = F.softmax(hidden, dim=-1)

        if target_tokens is not None:
            loss = F.cross_entropy(
                hidden.view(batch_size * seqlen, -1),
                target_tokens.view(batch_size * seqlen),
            )
            return new_logits, loss
        else:
            return new_logits

    def generate(self, text: str, dataLoader: DataLoader, max_new_token=50):
        tokens = dataLoader.text2token(text).unsqueeze(0)
        assert tokens.ndim == 2
        for _ in range(max_new_token):
            new_token = torch.multinomial(
                self(tokens[:, -1:]).squeeze(0)[-1], num_samples=1
            )
            tokens = torch.cat([tokens, new_token.unsqueeze_(0)], dim=-1)
        return [dataLoader.token2text(t) for t in tokens]
