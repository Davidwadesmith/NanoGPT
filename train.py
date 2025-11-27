import logging
import sys
from math import inf
import torch
import torch.nn as nn
from torch.optim import Optimizer
import torch.utils.tensorboard as tensorboard
from src import model
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

# Tensorboard Settings
train_writer = tensorboard.SummaryWriter(log_dir="./logs/nanoGPT/train")
val_writer = tensorboard.SummaryWriter(log_dir="./logs/nanoGPT/val")
forward_writer = tensorboard.SummaryWriter(log_dir="./logs/nanoGPT/forward")
learning_rate_writer = tensorboard.SummaryWriter(log_dir="./logs/nanoGPT/lr")
grad_writer = tensorboard.SummaryWriter(log_dir="./logs/nanoGPT/grad")
model_writer = tensorboard.SummaryWriter(log_dir="./logs/nanoGPT/model")


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
        for j in range(50):
            tokens, target_tokens = dataLoader.get_batch("train")
            _, loss = model(tokens, target_tokens)
            loss_sum += loss
            forward_writer.add_scalar("Loss/forward", loss, i * 50 + j)
            optimizer.zero_grad()
            loss.backward()
            learning_rate_writer.add_scalar(
                "Learning_rate", optimizer.param_groups[0]["lr"], i * 50 + j
            )
            optimizer.step()

        loss_avg = validate(model, dataLoader)
        if loss_avg < min_val_loss:
            torch.save(model.state_dict(), cfg.model_path + "final_loss.pt")
            min_val_loss = loss_avg

        for name, param in model.named_parameters():
            grad_writer.add_histogram(name, param.detach().cpu(), i)
            if param.grad is not None:
                grad_writer.add_histogram(name + "/grad", param.grad.detach().cpu(), i)
        train_writer.add_scalar("Loss/total_loss", loss_sum / 50, i)
        val_writer.add_scalar("Loss/total_loss", loss_avg, i)
        print(f"end of {i} epochs, train loss: {loss_sum / 50}, val loss: {loss_avg}")


if __name__ == "__main__":
    config = Config()
    dataLoader = DataLoader(config.dataset, config)
    transformer = Transformer(config, dataLoader.vocab_size)
    tf_optimizer = torch.optim.AdamW(transformer.parameters(), lr=config.learning_rate)
    try:
        train(transformer, tf_optimizer, dataLoader, config)
        train_writer.flush()
        val_writer.flush()
        forward_writer.flush()
        grad_writer.flush()
    except Exception as e:
        logger.error(f"Error in training loop: {type(e).__name__}", exc_info=True)

    dummyinput = torch.randint(
        0, dataLoader.vocab_size, (config.batch_size, config.seqlen)
    ).to(torch.device(config.device))
    model_writer.add_graph(transformer, dummyinput)
