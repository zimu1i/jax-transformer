# jax-transformer

Decoder-only Transformer built from scratch in JAX/Flax. No pre-built attention layers — the multi-head causal self-attention is hand-rolled in raw `jax.numpy` so I can derive every step on a whiteboard.

I'm building this to learn JAX properly (not just PyTorch-with-different-syntax) and to have something concrete to talk about in interviews. The goal is understanding, not scale.

## What's here

- **Scaled dot-product attention** with causal masking (`src/jaxtransformer/attention.py`)
- **Token + learned positional embeddings** (`src/jaxtransformer/embedding.py`)
- **Transformer block** — pre-norm layernorm, feed-forward MLP, residual connections (`src/jaxtransformer/block.py`)
- **Full model** wrapped in Flax modules with weight tying (`src/jaxtransformer/model.py`)
- **Tests** for shapes, causal masking, layer norm statistics, end-to-end causality, and gradient flow (`tests/`)

## Setup

```bash
pip install -e ".[dev]"
pytest
```

Requires Python 3.10+. Runs on CPU — no GPU needed.

## Status

Work in progress. Training loop and data pipeline are not implemented yet.
