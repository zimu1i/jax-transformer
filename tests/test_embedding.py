import jax
import jax.numpy as jnp

from jaxtransformer.embedding import embed


def test_embed_correctness():
    """embed(ids) must equal token_table[ids] + position_table[:seq_len].
    Catches off-by-one in position slicing or wrong gather logic."""
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)

    vocab_size, max_seq_len, d_model = 50, 32, 8
    tok_table = jax.random.normal(k1, (vocab_size, d_model))
    pos_table = jax.random.normal(k2, (max_seq_len, d_model))
    ids = jnp.array([[3, 7, 0], [49, 1, 20]])

    out = embed(ids, tok_table, pos_table)
    assert out.shape == (2, 3, d_model)

    for b in range(2):
        for s in range(3):
            expected = tok_table[ids[b, s]] + pos_table[s]
            assert jnp.allclose(out[b, s], expected), (
                f"Mismatch at batch={b}, pos={s}"
            )


def test_different_ids_different_embeddings():
    """Two sequences with different token IDs must produce different embeddings.
    Would fail if the lookup ignores the IDs entirely."""
    key = jax.random.PRNGKey(1)
    k1, k2 = jax.random.split(key)

    tok_table = jax.random.normal(k1, (100, 16))
    pos_table = jax.random.normal(k2, (32, 16))

    ids_a = jnp.array([[0, 1, 2, 3]])
    ids_b = jnp.array([[4, 5, 6, 7]])

    out_a = embed(ids_a, tok_table, pos_table)
    out_b = embed(ids_b, tok_table, pos_table)

    assert not jnp.allclose(out_a, out_b)
