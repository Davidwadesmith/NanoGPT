# NanoGPT - Pirate Version

本项目是 NanoGPT 的“海盗”版本，实现了一个简单易用的 GPT 语言模型。适合用于学习和实验生成式预训练变换器（GPT）的核心原理。

## 项目结构

主要代码均在 `src/` 目录下，包括：

- [`gpt.py`](https://github.com/Davidwadesmith/NanoGPT/blob/main/src/gpt.py)：核心 GPT 模型实现。
- [`eval.py`](https://github.com/Davidwadesmith/NanoGPT/blob/main/src/eval.py)：模型评估与推理脚本。
- [`input.txt`](https://github.com/Davidwadesmith/NanoGPT/blob/main/src/input.txt)：训练用语料文本。

## 快速开始

### 环境准备

python>=3.12，并根据需要自行安装 `torch` 等深度学习相关依赖。

### 训练模型

首先准备你的数据（替换或修改 `src/input.txt`），然后运行核心训练脚本：

```zsh
python src/gpt.py
```

模型会基于数据进行训练，详细参数和用法请查阅源码文档。

### 模型评估与文本生成

训练完成后，可使用评估脚本进行推理或生成文本：

```zsh
python src/eval.py
```

你可以根据需要修改脚本中的参数，生成不同风格的文本。

## 代码说明

- `gpt.py` 实现了 GPT 模型的完整训练与推理的基本逻辑，包括模型结构、损失函数等。
- `eval.py` 用于加载训练好的模型进行评估和生成。
- 项目全部用 Python 编写，适合熟悉 Python 和深度学习的用户快速上手和二次开发。

## 许可协议

本项目采用 MIT License。

