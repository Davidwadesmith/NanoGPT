import torch
from src.model import Transformer
from src.config import Config
from src.tokenizer import Tokenizer

if __name__ == "__main__":
    cfg = Config()
    tokenizer = Tokenizer("./data/meta.json")
    model = Transformer(cfg, tokenizer.vocab_size)
    state_dict = torch.load("./checkpoints/final_loss.pt", map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    text = """
    Rust in dust.
    """
    print(model.generate(text, tokenizer, max_new_token=500))
