"""
NanoGPT的训练以及类定义代码
"""
from math import inf
from typing import OrderedDict
from dataclasses import dataclass
import logging
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.optim import Optimizer

# ---Debug Settings
logging.basicConfig(
    level=logging.ERROR,
    format=logging.BASIC_FORMAT,
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


# ---global constant---
torch.manual_seed(114514)


@dataclass
class Config:
    """
    NanoGPT的各项配置参数
    """
    learning_rate: float = 1e-3
    epochs: int = 1000
    seqlen: int = 32
    batch_size: int = 64
    hidden_dim: int = 256
    head_n: int = 8
    block_n: int = 2
    device: str = "cpu"
    model_path: str = "../checkpoints/"
    dataset: str = "../data/input.txt"


class DataLoader:
    """
    加载dataset的类
    """
    def __init__(self, file: str, cfg: Config) -> None:
        self.cfg = cfg
        with open(file, "r", encoding="utf-8") as f:
            input_text = f.read()
        self.input_len = len(input_text)
        print(f"{file=} loaded, {self.input_len=}")
        self.validation_set = input_text[int(self.input_len * 0.9) :]
        self.training_set = input_text[: int(self.input_len * 0.9)]
        self.char_set = sorted(list(set(input_text)))
        self.vocab_size = len(self.char_set)
        self.char_map = {c: i for i, c in enumerate(self.char_set)}
        self.char_map_rev = {i: c for c, i in self.char_map.items()}
        logger.info(f"char_map: {self.char_map}")
        self.validation_tokens = self.text2token(self.validation_set)
        self.training_tokens = self.text2token(self.training_set)

    def get_batch(self, split: str):
        if split == "train":
            input_tokens = self.training_tokens
        elif split == "val":
            input_tokens = self.validation_tokens
        else:
            input_tokens = torch.tensor(None)

        rand_idx = self.rand_tokens(self.cfg.batch_size, self.cfg.seqlen, input_tokens)

        logger.debug(f"{rand_idx.shape=}")
        # offset = torch.arange(self.cfg.seqlen)
        # target_offset = torch.arange(1, self.cfg.seqlen + 1)
        # rand_idx.unsqueeze_(1)
        # idx_matrix = rand_idx + offset
        # target_idx_matrix = rand_idx + target_offset
        # tokens = input_tokens[idx_matrix]
        # target_tokens = input_tokens[target_idx_matrix]
        tokens = torch.stack(
            [input_tokens[idx : idx + self.cfg.seqlen] for idx in rand_idx]
        )
        target_tokens = torch.stack(
            [input_tokens[idx + 1 : idx + self.cfg.seqlen + 1] for idx in rand_idx]
        )
        # print(f"{tokens.shape=}")
        return tokens.to(torch.device(self.cfg.device)), target_tokens.to(
            torch.device(self.cfg.device)
        )

    def text2token(self, text: str):
        return torch.tensor([self.char_map[char] for char in text])

    def token2text(self, tokens: torch.Tensor):
        return [self.char_map_rev[int(token.item())] for token in tokens]

    def rand_tokens(self, batch_size: int, seqlen: int, tokens: torch.Tensor):
        input_len = len(tokens)
        assert input_len - seqlen > 0
        return torch.randint(0, input_len - seqlen, (batch_size,))


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

    def forward(self, hidden_tokens):
        q = torch.transpose(
            self.wq(hidden_tokens).view(
                self.cfg.batch_size,
                self.cfg.seqlen,
                self.cfg.head_n,
                self.cfg.hidden_dim // self.cfg.head_n,
            ),
            1,
            2,
        )
        k = torch.transpose(
            self.wk(hidden_tokens).view(
                self.cfg.batch_size,
                self.cfg.seqlen,
                self.cfg.head_n,
                self.cfg.hidden_dim // self.cfg.head_n,
            ),
            1,
            2,
        )
        v = torch.transpose(
            self.wv(hidden_tokens).view(
                self.cfg.batch_size,
                self.cfg.seqlen,
                self.cfg.head_n,
                self.cfg.hidden_dim // self.cfg.head_n,
            ),
            1,
            2,
        )
        attention_map = (
            q
            @ torch.transpose(k, -2, -1)
            / ((self.cfg.hidden_dim // self.cfg.head_n) ** 0.5)
        )

        self.mask = (torch.tril(torch.ones(self.cfg.seqlen, self.cfg.seqlen)) == 0).to(
            torch.device(self.cfg.device)
        )

        masked_attention_map = torch.masked_fill(attention_map, self.mask, -inf)
        masked_attention = self.softmax(masked_attention_map) @ v

        return torch.transpose(masked_attention, 1, 2).reshape(
            self.cfg.batch_size, self.cfg.seqlen, self.cfg.hidden_dim
        )


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

    def forward(self, hidden_tokens: torch.Tensor):
        return self.fastforward_layer_out(
            self.gelu(self.fastforward_layer_in(hidden_tokens))
        )


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

    def forward(self, hidden_tokens: torch.Tensor):
        mean = hidden_tokens.mean(dim=-1, keepdim=True)
        variance = torch.var(hidden_tokens, dim=-1, keepdim=True)
        x = (hidden_tokens - mean) / (variance**0.5)
        y = (self.gamma * x) / ((variance + 1e-5) ** 0.5) + self.beta
        return y


class Block(nn.Module):
    def __init__(self, cfg: Config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.multiheadAttention = MultiheadAttention(cfg)
        self.layernorm_1 = LayerNorm(cfg)
        self.layernorm_2 = LayerNorm(cfg)
        self.ffn = FFN(cfg)

    def forward(self, hidden_tokens: torch.Tensor):
        x = self.multiheadAttention(self.layernorm_1(hidden_tokens)) + hidden_tokens
        return self.ffn(self.layernorm_2(x)) + x


class Transformer(nn.Module):
    def __init__(
        self,
        cfg: Config = Config(),
        vocab_size: int = len(DataLoader(Config().dataset, Config()).char_set),
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

    def forward(self, tokens, target_tokens=None):
        tokens = tokens.to(torch.device(self.cfg.device))
        if target_tokens is not None:
            target_tokens = target_tokens.to(torch.device(self.cfg.device))

        batch_size = len(tokens)
        seqlen = len(tokens[0])

        logger.debug(f"{tokens.shape=}")

        embeddings = self.embedding_layer(tokens)

        self.positional_embedding_layer = self.PositionEmbedding(
            self.cfg.seqlen, self.cfg.hidden_dim
        ).to(torch.device(self.cfg.device))

        embeddings = embeddings + self.positional_embedding_layer
        logger.debug(f"{embeddings.shape=}")

        hidden = self.blocks(embeddings)
        last = self.linear(hidden)

        # print(embeddings.shape)
        new_logits = F.softmax(last, dim=-1)

        if target_tokens is not None:
            loss = F.cross_entropy(
                last.view(batch_size * seqlen, -1),
                target_tokens.view(batch_size * seqlen),
            )
            return new_logits, loss
        else:
            return new_logits

    def generate(self, text: str, dataLoader: DataLoader, max_new_token=50):
        logger.critical(f"{text=}")
        cfg = self.cfg
        tmp_batch_size = cfg.batch_size
        cfg.batch_size = 1
        self.cfg = cfg

        tokens = (
            dataLoader.text2token(text).unsqueeze(0).to(torch.device(self.cfg.device))
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
        return [dataLoader.token2text(t) for t in tokens]

    def PositionEmbedding(self, seqlen: int, hidden_dim: int):
        position = torch.arange(seqlen).unsqueeze(1)
        pe = torch.zeros(seqlen, hidden_dim)
        div_term = torch.exp(
            -torch.arange(0, hidden_dim, 2)
            / hidden_dim
            * torch.log(torch.tensor(10000))
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


@torch.no_grad
def validate(model: nn.Module, dataLoader: DataLoader):
    """
    helper function for evaluate loss on validation set
    """
    sum = 0
    for _ in range(10):
        tokens, target_tokens = dataLoader.get_batch("val")
        _, loss = model(tokens, target_tokens)
        sum += loss

    return sum / 10


def train(model: nn.Module, optimizer: Optimizer, dataLoader: DataLoader, cfg: Config):
    """
    training function
    """
    min_val_loss = inf
    for i in range(cfg.epochs):
        loss_sum = 0
        for _ in range(10):
            tokens, target_tokens = dataLoader.get_batch("train")
            _, loss = model(tokens, target_tokens)
            loss_sum += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_avg = validate(model, dataLoader)
        if loss_avg < min_val_loss:
            torch.save(model.state_dict(), cfg.model_path + "final_loss.pt")
            min_val_loss = loss_avg
        print(f"end of {i} epochs, train loss: {loss_sum / 10}, val loss: {loss_avg}")


if __name__ == "__main__":
    config = Config()
    dataLoader = DataLoader(config.dataset, config)
    model = BigramModel(config, dataLoader.vocab_size, hidden_dim=256)
    transformer = Transformer(config, dataLoader.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    tf_optimizer = torch.optim.AdamW(transformer.parameters(), lr=config.learning_rate)
    text = """
COUNTESS
In delivering my son from me, I bury a second husband.
BERTRAM
And I in going, madam, weep o'er my father's death
anew: but I must attend his majesty's command, to
whom I am now in ward, evermore in subjection.
LAFEU
You shall find of the king a husband, madam; you,
sir, a father: he that so generally is at all times
good must of necessity hold his virtue to you; whose
worthiness would stir it up where it wanted rather
than lack it where there is such abundance.
COUNTESS
What hope is there of his majesty's amendment?
LAFEU
He hath abandoned his physicians, madam; under whose
practises he hath persecuted time with hope, and
finds no other advantage in the process but only the
losing of hope by time.
COUNTESS
This young gentlewoman had a father,--O, that
'had'! how sad a passage 'tis!--whose skill was
almost as great as his honesty; had it stretched so
far, would have made nature immortal, and death
should have play for lack of work. Would, for the
king's sake, he were living! I think it would be
the death of the king's disease.
LAFEU
How called you the man you speak of, madam?
COUNTESS
He was famous, sir, in his profession, and it was
his great right to be so: Gerard de Narbon.
LAFEU
He was excellent indeed, madam: the king very
lately spoke of him admiringly and mourningly: he
was skilful enough to have lived still, if knowledge
could be set up against mortality.
    """

    print("".join(transformer.generate(text, dataLoader, max_new_token=90)[0]))
    try:
        train(transformer, tf_optimizer, dataLoader, config)
    except Exception as e:
        logger.error(f"Error in training loop: {type(e).__name__}", exc_info=True)
    print("".join(transformer.generate(text, dataLoader, max_new_token=90)[0]))

    # print("".join(model.generate(text, dataLoader, max_new_token=90)[0]))
    # train(model, dataLoader, config)
    # print("".join(model.generate(text, dataLoader, max_new_token=90)[0]))
