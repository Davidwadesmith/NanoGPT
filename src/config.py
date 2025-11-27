"""
Config
"""

from dataclasses import dataclass


@dataclass
class Config:
    """
    NanoGPT的各项配置参数
    """

    learning_rate: float = 1e-3
    epochs: int = 1000
    vocab_size: int = 0
    seqlen: int = 32
    batch_size: int = 64
    hidden_dim: int = 256
    head_n: int = 8
    block_n: int = 2
    device: str = "cpu"
    model_path: str = "./checkpoints/"
    dataset: str = "./data/input.txt"
