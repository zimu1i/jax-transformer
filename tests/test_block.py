import jax
import jax.numpy as jnp

from jaxtransformer.block import transformer_block, layer_norm


def _block_inputs(seed=0, batch=2, seq_len=8, d_model=16, num_heads=4):
    """Generate random input and all params for one transformer block."""
    d_ff = d_model * 4
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 8)

    x = jax.random.normal(keys[0], (batch, seq_len, d_model))

    ln1_gamma = jnp.ones(d_model)
    ln1_beta = jnp.zeros(d_model)
    ln2_gamma = jnp.ones(d_model)
    ln2_beta = jnp.zeros(d_model)

    w_q = jax.random.normal(keys[1], (d_model, d_model)) * 0.02
    w_k = jax.random.normal(keys[2], (d_model, d_model)) * 0.02
    w_v = jax.random.normal(keys[3], (d_model, d_model)) * 0.02
    w_o = jax.random.normal(keys[4], (d_model, d_model)) * 0.02

    ff_w1 = jax.random.normal(keys[5], (d_model, d_ff)) * 0.02
    ff_b1 = jnp.zeros(d_ff)
    ff_w2 = jax.random.normal(keys[6], (d_ff, d_model)) * 0.02
    ff_b2 = jnp.zeros(d_model)

    return (x, ln1_gamma, ln1_beta, w_q, w_k, w_v, w_o,
            ln2_gamma, ln2_beta, ff_w1, ff_b1, ff_w2, ff_b2, num_heads)


def test_block_output_shape():
    """Block must preserve (batch, seq_len, d_model)."""
    args = _block_inputs()
    out, weights = transformer_block(*args)
    assert out.shape == args[0].shape
    assert weights.shape == (2, 4, 8, 8)


def test_block_residual_connection():
    """Output should differ from input (sublayers contribute), but the
    difference should be small with small init — confirming the residual
    carries the input through."""
    args = _block_inputs()
    x = args[0]
    out, _ = transformer_block(*args)

    assert not jnp.allclose(out, x), "Output identical to input — sublayers inactive"
    diff = jnp.abs(out - x).mean()
    assert diff < 1.0, f"Residual correction too large ({diff:.2f}) — something wrong"


def test_block_gradients_finite():
    """Gradients must flow through layernorm, attention, residual, FFN, and
    second residual without dying or exploding."""
    args = _block_inputs()

    def loss_fn(*params):
        out, _ = transformer_block(*params)
        return out.sum()

    grads = jax.grad(loss_fn, argnums=tuple(range(len(args) - 1)))(*args)

    for i, g in enumerate(grads):
        assert jnp.all(jnp.isfinite(g)), f"Gradient {i} contains inf/NaN"


def test_layer_norm_statistics():
    """After layer norm, each vector should have mean ~0 and variance ~1
    (before gamma/beta scaling). Verifies the normalization math."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, 4, 32)) * 5 + 3  # offset mean and scale

    gamma = jnp.ones(32)
    beta = jnp.zeros(32)
    normed = layer_norm(x, gamma, beta)

    means = normed.mean(axis=-1)
    vars_ = normed.var(axis=-1)

    assert jnp.allclose(means, 0.0, atol=1e-5), f"Mean not ~0: {means}"
    assert jnp.allclose(vars_, 1.0, atol=1e-3), f"Variance not ~1: {vars_}"
