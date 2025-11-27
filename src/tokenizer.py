import json
import torch


class Tokenizer:
    def __init__(self, file: str) -> None:
        with open(file, "r", encoding="utf-8") as f:
            self.char_map = json.load(f)
            self.char_map_rev = {i: c for c, i in self.char_map.items()}
            self.vocab_size = len(self.char_map)

    def text2token(self, text: str):
        return torch.tensor([self.char_map[char] for char in text])

    def token2text(self, tokens: torch.Tensor):
        return [self.char_map_rev[int(token.item())] for token in tokens]
