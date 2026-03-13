import json

import torch


class Tokenizer:
    def __init__(self, file: str) -> None:
        with open(file, "r", encoding="utf-8") as f:
            meta = json.load(f)
            self.char_map = meta["char to id"]
            # JSON object keys are strings; normalize to int for token lookup.
            self.char_map_rev = {int(k): v for k, v in meta["id to char"].items()}
            self.vocab_size = meta["vocab size"]
            self.eos_id = self._resolve_eos_id(meta)

    def _resolve_eos_id(self, meta: dict) -> int | None:
        eos_id = meta.get("eos id")
        if eos_id is not None:
            return int(eos_id)

        eos_token = meta.get("eos token")
        if eos_token is not None and eos_token in self.char_map:
            return int(self.char_map[eos_token])

        for candidate in ("<eos>", "</s>", "[EOS]", "<|endoftext|>"):
            if candidate in self.char_map:
                return int(self.char_map[candidate])

        return None

    def text2token(self, text: str):
        return torch.tensor([self.char_map[char] for char in text])

    def token2text(self, tokens: torch.Tensor):
        return [self.char_map_rev.get(int(token.item()), "") for token in tokens]
