import logging
import torch
import json
from .config import Config

logger = logging.getLogger(__name__)


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

        with open("./data/meta.json", "w", encoding="utf-8") as f:
            json.dump(self.char_map, f)

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
        tokens = torch.stack(
            [input_tokens[idx : idx + self.cfg.seqlen] for idx in rand_idx]
        )
        target_tokens = torch.stack(
            [input_tokens[idx + 1 : idx + self.cfg.seqlen + 1] for idx in rand_idx]
        )
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
