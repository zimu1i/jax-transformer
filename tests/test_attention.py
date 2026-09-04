import jax
import jax.numpy as jnp
import pytest

from jaxtransformer.attention import multihead_causal_self_attention, make_causal_mask


# --- Helpers ---

def _random_inputs(seed=0, batch=2, seq_len=8, d_model=16, num_heads=4):
    """Generate random input and weight matrices for testing."""
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 5)
    x   = jax.random.normal(keys[0], (batch, seq_len, d_model))
    w_q = jax.random.normal(keys[1], (d_model, d_model)) * 0.02
    w_k = jax.random.normal(keys[2], (d_model, d_model)) * 0.02
    w_v = jax.random.normal(keys[3], (d_model, d_model)) * 0.02
    w_o = jax.random.normal(keys[4], (d_model, d_model)) * 0.02
    return x, w_q, w_k, w_v, w_o, num_heads


# --- Test 1: shapes ---

@pytest.mark.parametrize("batch,seq_len,d_model,num_heads", [
    (1, 4, 8, 2),
    (2, 8, 16, 4),
    (3, 1, 32, 8),   # seq_len=1: a single token (first step of generation)
])
def test_output_shapes(batch, seq_len, d_model, num_heads):
    """Output must preserve (batch, seq_len, d_model); weights must be
    (batch, num_heads, seq_len, seq_len)."""
    x, w_q, w_k, w_v, w_o, _ = _random_inputs(
        batch=batch, seq_len=seq_len, d_model=d_model, num_heads=num_heads,
    )
    output, weights = multihead_causal_self_attention(x, w_q, w_k, w_v, w_o, num_heads)

    assert output.shape == (batch, seq_len, d_model)
    assert weights.shape == (batch, num_heads, seq_len, seq_len)


# --- Test 2: causal mask blocks future positions ---

def test_causal_mask_zeros():
    """Every attention weight where key position > query position must be zero.
    This is the defining property of decoder self-attention."""
    x, w_q, w_k, w_v, w_o, num_heads = _random_inputs(seq_len=8)
    _, weights = multihead_causal_self_attention(x, w_q, w_k, w_v, w_o, num_heads)

    seq_len = weights.shape[-1]
    # Upper triangle (k=1 excludes the diagonal): these are future positions.
    upper = jnp.triu(weights, k=1)
    assert jnp.allclose(upper, 0.0), (
        f"Future positions have non-zero attention: max = {upper.max()}"
    )


def test_causal_mask_rows_sum_to_one():
    """Each query's weights over allowed keys must form a valid distribution."""
    x, w_q, w_k, w_v, w_o, num_heads = _random_inputs(seq_len=8)
    _, weights = multihead_causal_self_attention(x, w_q, w_k, w_v, w_o, num_heads)

    row_sums = weights.sum(axis=-1)
    assert jnp.allclose(row_sums, 1.0, atol=1e-5), (
        f"Row sums deviate from 1.0: min={row_sums.min()}, max={row_sums.max()}"
    )


# --- Test 3: gradients exist and are finite ---

def test_gradients_finite():
    """Backprop through attention must produce finite, non-zero gradients for
    every parameter. Catches NaN from bad masking, inf from unscaled logits,
    and zero from accidentally detached paths."""
    x, w_q, w_k, w_v, w_o, num_heads = _random_inputs()

    def loss_fn(x, w_q, w_k, w_v, w_o):
        output, _ = multihead_causal_self_attention(x, w_q, w_k, w_v, w_o, num_heads)
        return output.sum()

    grads = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4))(x, w_q, w_k, w_v, w_o)

    param_names = ["x", "w_q", "w_k", "w_v", "w_o"]
    for name, g in zip(param_names, grads):
        assert jnp.all(jnp.isfinite(g)), f"Gradient for {name} contains inf or NaN"
        assert not jnp.allclose(g, 0.0), f"Gradient for {name} is all zeros"
