import torch

from src.config import Config
from src.model import Transformer
from src.tokenizer import Tokenizer

if __name__ == "__main__":
    cfg = Config()
    tokenizer = Tokenizer("./data/meta.json")
    state_dict = torch.load("./checkpoints/final_loss.pt", map_location="cpu")
    checkpoint_vocab_size = state_dict["embedding_layer.weight"].shape[0]
    if checkpoint_vocab_size != tokenizer.vocab_size:
        raise ValueError(
            "Checkpoint vocab size does not match meta.json. "
            "After adding a dedicated <eos> token, rerun src/prepare.py and retrain to get a compatible checkpoint."
        )
    model = Transformer(cfg, tokenizer.vocab_size)
    model.load_state_dict(state_dict, strict=True)
    text = "Once upon a time."
    print(model.generate(text, tokenizer, max_new_token=500))
