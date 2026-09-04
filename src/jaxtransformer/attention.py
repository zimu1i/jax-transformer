import jax
import jax.numpy as jnp


def scaled_dot_product_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Scaled dot-product attention with optional mask.

    Operates over arbitrary leading batch dimensions — works for both
    (batch, seq, d_k) and (batch, num_heads, seq, d_k).

    Args:
        q: (..., seq_len, d_k)
        k: (..., seq_len, d_k)
        v: (..., seq_len, d_v)
        mask: Broadcastable boolean mask. True = attend, False = block.

    Returns:
        output: (..., seq_len, d_v)
        weights: (..., seq_len, seq_len)
    """
    d_k = q.shape[-1]

    attn_logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1))
    attn_logits = attn_logits / jnp.sqrt(d_k).astype(q.dtype)

    if mask is not None:
        attn_logits = jnp.where(mask, attn_logits, -1e9)

    weights = jax.nn.softmax(attn_logits, axis=-1)
    output = jnp.matmul(weights, v)

    return output, weights


def make_causal_mask(seq_len: int) -> jax.Array:
    """Lower-triangular boolean mask: position i can attend to positions 0..i.

    Returns:
        mask: shape (seq_len, seq_len), dtype bool.
    """
    return jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))


def multihead_causal_self_attention(
    x: jax.Array,
    w_q: jax.Array,
    w_k: jax.Array,
    w_v: jax.Array,
    w_o: jax.Array,
    num_heads: int,
) -> tuple[jax.Array, jax.Array]:
    """Multi-head causal self-attention from scratch.

    Args:
        x:     Input, shape (batch, seq_len, d_model).
        w_q:   Query projection,  shape (d_model, d_model).
        w_k:   Key projection,    shape (d_model, d_model).
        w_v:   Value projection,  shape (d_model, d_model).
        w_o:   Output projection, shape (d_model, d_model).
        num_heads: Number of attention heads. Must divide d_model evenly.

    Returns:
        output:  shape (batch, seq_len, d_model)
        weights: shape (batch, num_heads, seq_len, seq_len)
    """
    batch, seq_len, d_model = x.shape
    head_dim = d_model // num_heads

    # --- Step 1: project into Q, K, V ---
    # (batch, seq_len, d_model) @ (d_model, d_model) -> (batch, seq_len, d_model)
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v

    # --- Step 2: split into heads ---
    # Reshape: (batch, seq_len, d_model) -> (batch, seq_len, num_heads, head_dim)
    # Transpose: -> (batch, num_heads, seq_len, head_dim)
    # Now num_heads is a batch dimension for our attention function.
    q = q.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(batch, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)

    # --- Step 3: attention with causal mask ---
    # Mask shape (seq_len, seq_len) broadcasts over (batch, num_heads, ...).
    mask = make_causal_mask(seq_len)
    output, weights = scaled_dot_product_attention(q, k, v, mask=mask)

    # --- Step 4: concatenate heads ---
    # Transpose: (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim)
    # Reshape:   -> (batch, seq_len, d_model)
    output = output.transpose(0, 2, 1, 3).reshape(batch, seq_len, d_model)

    # --- Step 5: output projection ---
    # (batch, seq_len, d_model) @ (d_model, d_model) -> (batch, seq_len, d_model)
    output = output @ w_o

    return output, weights
