import jax
import jax.numpy as jnp

from jaxtransformer.attention import multihead_causal_self_attention


def layer_norm(
    x: jax.Array,
    gamma: jax.Array,
    beta: jax.Array,
    eps: float = 1e-5,
) -> jax.Array:
    """Layer normalization over the last axis (d_model).

    Args:
        x:     (..., d_model)
        gamma: (d_model,) learned scale.
        beta:  (d_model,) learned shift.
        eps:   Small constant for numerical stability.

    Returns:
        (..., d_model)
    """
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / jnp.sqrt(var + eps)
    return gamma * x_norm + beta


def feed_forward(
    x: jax.Array,
    w1: jax.Array,
    b1: jax.Array,
    w2: jax.Array,
    b2: jax.Array,
) -> jax.Array:
    """Position-wise feed-forward network: expand, GELU, project back.

    Args:
        x:  (batch, seq_len, d_model)
        w1: (d_model, d_ff)    — expand
        b1: (d_ff,)
        w2: (d_ff, d_model)    — contract
        b2: (d_model,)

    Returns:
        (batch, seq_len, d_model)
    """
    hidden = jax.nn.gelu(x @ w1 + b1)
    return hidden @ w2 + b2


def transformer_block(
    x: jax.Array,
    ln1_gamma: jax.Array,
    ln1_beta: jax.Array,
    w_q: jax.Array,
    w_k: jax.Array,
    w_v: jax.Array,
    w_o: jax.Array,
    ln2_gamma: jax.Array,
    ln2_beta: jax.Array,
    ff_w1: jax.Array,
    ff_b1: jax.Array,
    ff_w2: jax.Array,
    ff_b2: jax.Array,
    num_heads: int,
) -> tuple[jax.Array, jax.Array]:
    """One pre-norm Transformer decoder block.

    Flow:
        residual ─────────────────── add ─────────────────── add ─→ out
                 ↘                  ↗     ↘                  ↗
              layernorm → MH-attention   layernorm → FFN

    Args:
        x: (batch, seq_len, d_model)
        ln1_*: LayerNorm params for the attention sub-layer.
        w_q/k/v/o: Attention projection weights.
        ln2_*: LayerNorm params for the feed-forward sub-layer.
        ff_*: Feed-forward weights and biases.
        num_heads: Number of attention heads.

    Returns:
        output:  (batch, seq_len, d_model)
        weights: (batch, num_heads, seq_len, seq_len)
    """
    # --- Sub-layer 1: multi-head causal self-attention ---
    normed = layer_norm(x, ln1_gamma, ln1_beta)
    attn_out, weights = multihead_causal_self_attention(
        normed, w_q, w_k, w_v, w_o, num_heads,
    )
    x = x + attn_out

    # --- Sub-layer 2: position-wise feed-forward ---
    normed = layer_norm(x, ln2_gamma, ln2_beta)
    ff_out = feed_forward(normed, ff_w1, ff_b1, ff_w2, ff_b2)
    x = x + ff_out

    return x, weights
