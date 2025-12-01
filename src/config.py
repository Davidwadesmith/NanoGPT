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
    epochs: int = 20
    vocab_size: int = 0
    seqlen: int = 32
    batch_size: int = 32
    hidden_dim: int = 128
    head_n: int = 8
    block_n: int = 2
    device: str = "cpu"
    model_path: str = "./checkpoints/"
    dataset: str = "./data/input.txt"

    embed: str = "RoPE"
    norm: str = "RMSNorm"
