from gpt import DataLoader, Transformer, Config
import torch

if __name__ == "__main__":
    cfg = Config()
    model = Transformer()
    state_dict = torch.load("../checkpoints/final_loss.pt", map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    text = """
    Rust in dust.
    """
    print(
        "".join(
            model.generate(
                text, DataLoader(Config().dataset, Config()), max_new_token=500
            )[0]
        )
    )
