# jax-transformer

Decoder-only Transformer built from scratch in JAX/Flax. No pre-built attention layers. The multi-head causal self-attention is hand-rolled in raw `jax.numpy` so I can derive every step on a whiteboard.

I'm building this to learn JAX properly (not just PyTorch-with-different-syntax) and to have something concrete to talk about in interviews. The goal is understanding, not scale.

## Content

- **Scaled dot-product attention** with causal masking (`src/jaxtransformer/attention.py`)
- **Token + learned positional embeddings** (`src/jaxtransformer/embedding.py`)
- **Transformer block** with pre-norm layernorm, feed-forward MLP, residual connections (`src/jaxtransformer/block.py`)
- **Full model** wrapped in Flax modules with weight tying (`src/jaxtransformer/model.py`)
- **Character tokenizer and batching** (`src/jaxtransformer/data.py`)
- **Training loop** with cross-entropy loss, Adam optimizer, and autoregressive generation (`src/jaxtransformer/train.py`)
- **Tests** for shapes, causal masking, layer norm statistics, end-to-end causality, and gradient flow (`tests/`)

## Setup

```bash
pip install -e ".[dev]"
pytest
```

Requires Python 3.10+. Runs on CPU, no GPU needed.

## Training

Train on a text file:

```bash
python train_example.py data.txt --epochs 10
```

Or run with the built-in example text:

```bash
python train_example.py
```

See `python train_example.py --help` for all options (model size, learning rate, etc).
