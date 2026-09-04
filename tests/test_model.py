import jax
import jax.numpy as jnp

from jaxtransformer.model import DecoderTransformer


def _make_model_and_params(seed=0):
    """Small model for testing."""
    model = DecoderTransformer(
        vocab_size=64,
        max_seq_len=32,
        d_model=16,
        num_heads=4,
        num_layers=2,
        d_ff=64,
    )
    key = jax.random.PRNGKey(seed)
    token_ids = jax.random.randint(jax.random.PRNGKey(seed + 1), (2, 8), 0, 64)
    params = model.init(key, token_ids)
    return model, params, token_ids


def test_logits_shape():
    """Final output must be (batch, seq_len, vocab_size) — the model's contract."""
    model, params, token_ids = _make_model_and_params()
    logits = model.apply(params, token_ids)
    assert logits.shape == (2, 8, 64)


def test_causality_end_to_end():
    """Changing a token at position t must NOT affect logits at positions 0..t-1.

    This is the strongest causality test: it verifies the property through the
    entire model (embeddings, every block, final norm, output projection),
    not just the attention mask. If any layer leaks future information, this
    catches it."""
    model, params, token_ids = _make_model_and_params()

    logits_original = model.apply(params, token_ids)

    # Modify the last token (position 7)
    modified = token_ids.at[0, 7].set((token_ids[0, 7] + 1) % 64)
    logits_modified = model.apply(params, modified)

    # Positions 0-6 must be unchanged for batch element 0
    assert jnp.allclose(logits_original[0, :7], logits_modified[0, :7]), (
        "Changing token 7 affected earlier positions — causality violation"
    )
    # Position 7 should change (sanity check that the modification did something)
    assert not jnp.allclose(logits_original[0, 7], logits_modified[0, 7]), (
        "Changing token 7 had no effect on position 7 — model ignores input?"
    )


def test_model_gradients_finite():
    """Backprop from logits through every layer to every parameter must produce
    finite gradients. Catches vanishing/exploding gradients in the full stack."""
    model, params, token_ids = _make_model_and_params()

    def loss_fn(params):
        logits = model.apply(params, token_ids)
        return logits.sum()

    grads = jax.grad(loss_fn)(params)

    for path, leaf in jax.tree_util.tree_leaves_with_path(grads):
        name = "/".join(str(k.key) for k in path)
        assert jnp.all(jnp.isfinite(leaf)), f"Gradient for {name} has inf/NaN"
        assert not jnp.allclose(leaf, 0.0), f"Gradient for {name} is all zeros"
