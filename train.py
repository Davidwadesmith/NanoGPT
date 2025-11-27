import logging
import sys
from math import inf
import torch
import torch.nn as nn
from torch.optim import Optimizer
from src.model import Transformer
from src.dataloader import DataLoader
from src.config import Config

# Debug Settings
logging.basicConfig(
    level=logging.ERROR,
    format=logging.BASIC_FORMAT,
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)


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
        for _ in range(50):
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
        print(f"end of {i} epochs, train loss: {loss_sum / 50}, val loss: {loss_avg}")


if __name__ == "__main__":
    config = Config()
    dataLoader = DataLoader(config.dataset, config)
    transformer = Transformer(config, dataLoader.vocab_size)
    tf_optimizer = torch.optim.AdamW(transformer.parameters(), lr=config.learning_rate)
    try:
        train(transformer, tf_optimizer, dataLoader, config)
    except Exception as e:
        logger.error(f"Error in training loop: {type(e).__name__}", exc_info=True)
