import json
import logging

import torch
import numpy as np

from .config import Config

logger = logging.getLogger(__name__)


class DataLoader:
    """
    加载dataset的类
    """

    def __init__(self, train_file: str, validate_file: str, cfg: Config) -> None:
        self.cfg = cfg
        meta = json.load(open(cfg.meta_data_path, "r", encoding="utf-8"))
        self.char_map = meta["char to id"]
        self.char_map_rev = {int(k): v for k, v in meta["id to char"].items()}
        self.vocab_size = meta["vocab size"]
        self.eos_id = meta.get("eos id")
        self.training_data = np.memmap(
            train_file, dtype=np.int32, mode="r"
        )
        self.validation_data = np.memmap(
            validate_file, dtype=np.int32, mode="r"
        )         

    def get_batch(self, split: str):
        if split == "train":
            input_data = self.training_data
        elif split == "val":
            input_data = self.validation_data
        else:
            raise ValueError("split must be 'train' or 'val'")

        rand_idx = self.rand_tokens(
            self.cfg.batch_size, self.cfg.seqlen, len(input_data)
        )

        logger.debug(f"{rand_idx.shape=}")
        tokens = torch.stack(
            [
                torch.tensor(input_data[idx : idx + self.cfg.seqlen], dtype=torch.long)
                for idx in rand_idx
            ]
        )
        target_tokens = torch.stack(
            [
                torch.tensor(
                    input_data[idx + 1 : idx + self.cfg.seqlen + 1], dtype=torch.long
                )
                for idx in rand_idx
            ]
        )
        return tokens.to(torch.device(self.cfg.device)), target_tokens.to(
            torch.device(self.cfg.device)
        )

    def text2token(self, text: str):
        return torch.tensor([self.char_map[char] for char in text])

    def token2text(self, tokens: torch.Tensor):
        return [self.char_map_rev[int(token.item())] for token in tokens]

    def rand_tokens(self, batch_size: int, seqlen: int, input_len: int):
        assert input_len - seqlen > 0
        return torch.randint(0, input_len - seqlen, (batch_size,))
