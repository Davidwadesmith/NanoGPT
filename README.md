# NanoGPT

本项目参照 [NanoGPT](https://github.com/karpathy/nanoGPT) ，实现了一个简单易用的 GPT 语言模型。适合用于学习和实验生成式预训练变换器（GPT）的核心原理。

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

## 🗺️ Roadmap & Learning Path

这是一个以 NanoGPT 为起点的深度学习科研训练路线图。本项目旨在从零开始理解 LLM 的每一个细节，并逐步将其改造为现代化的、支持科研实验的高性能框架。

### Phase 1: 夯实基础与工程化 (Foundation & Engineering)
*目标：彻底理解 Transformer 代码，并构建一个鲁棒的实验框架。*
- [ ] **代码重构与注释**：对 `gpt.py` 进行逐行注释，绘制数据流图；添加 Type Hinting 增强代码可读性。
- [ ] **单元测试 (Unit Tests)**：为 Attention、FeedForward 等模块编写测试用例，确保形状（Shape）变换正确。
- [ ] **可视化监控**：接入 WandB 或 TensorBoard，监控 Loss、Grad Norm、Learning Rate 变化，学会通过曲线诊断训练问题。
- [ ] **HuggingFace 兼容**：编写脚本支持加载/导出 HuggingFace 格式权重，方便利用社区生态进行评估。

### Phase 2: 架构现代化 "Llama-fication" (Modern Architecture)
*目标：将古老的 GPT-2 架构升级为现代 LLM (如 Llama 3/Gemma) 的主流架构，理解每个组件的数学原理。*
- [ ] **位置编码升级**：移除绝对位置编码，实现 **RoPE (Rotary Positional Embeddings)**。
- [ ] **归一化升级**：将 LayerNorm 替换为 **RMSNorm**，并尝试 Pre-Norm 架构。
- [ ] **激活函数升级**：将 GELU 替换为 **SwiGLU**。
- [ ] **注意力机制优化**：实现 **GQA (Grouped Query Attention)**，理解 KV Cache 的显存优化原理。
- [ ] **权重初始化**：研究并复现不同的初始化策略（如 MuP），观察对收敛速度的影响。

### Phase 3: 高性能计算与系统优化 (HPC & Efficiency)
*目标：深入 PyTorch 底层与 GPU 编程，提升训练与推理效率。*
- [ ] **算子融合**：集成 **Flash Attention 2**，对比手动实现 Attention 的速度差异。
- [ ] **混合精度训练**：完善 `bfloat16` 训练流程，理解精度溢出与数值稳定性问题。
- [ ] **分布式训练基础**：从 DDP (DistributedDataParallel) 进阶到初步理解 FSDP (Fully Sharded Data Parallel)。
- [ ] **自定义算子 (Optional)**：尝试用 Triton 或 CUDA 编写一个简单的 LayerNorm 或 Softmax 算子，理解 Kernel 优化。

### Phase 4: 前沿科研探索 (Research Frontier Playground)
*目标：复现经典/最新论文，作为科研练手。*
- [ ] **长窗口扩展**：尝试实现 ALiBi 或 YaRN 等长上下文技术。
- [ ] **稀疏化模型**：实现简单的 **MoE (Mixture of Experts)** 架构。
- [ ] **推测解码 (Speculative Decoding)**：利用小模型辅助大模型加速推理，编写完整的 draft-verify 循环。
- [ ] **参数高效微调 (PEFT)**：手动实现 LoRA (Low-Rank Adaptation)，而不是直接调库，理解其梯度更新逻辑。

### Phase 5: 数据与对齐 (Data & Alignment)
- [ ] **Tokenizer 深入**：训练自己的 BPE Tokenizer，对比不同词表大小对压缩率的影响。
- [ ] **指令微调 (SFT)**：构建简单的指令数据集，实现 Chat 模式。
- [ ] **偏好对齐**：尝试实现 DPO (Direct Preference Optimization)，理解 RLHF 的简化版本。

## 许可协议

本项目采用 MIT License。

