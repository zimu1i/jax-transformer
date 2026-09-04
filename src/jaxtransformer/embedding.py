import jax
import jax.numpy as jnp


def embed_tokens(token_ids: jax.Array, embedding_table: jax.Array) -> jax.Array:
    """Look up token embeddings by index.

    Args:
        token_ids:       (batch, seq_len), integer token IDs.
        embedding_table: (vocab_size, d_model), learned embedding matrix.

    Returns:
        (batch, seq_len, d_model)
    """
    return embedding_table[token_ids]


def embed_positions(seq_len: int, position_table: jax.Array) -> jax.Array:
    """Look up learned positional embeddings for positions 0..seq_len-1.

    Args:
        seq_len:        Number of positions to embed.
        position_table: (max_seq_len, d_model), learned position matrix.

    Returns:
        (seq_len, d_model) — broadcasts over batch when added.
    """
    return position_table[:seq_len]


def embed(
    token_ids: jax.Array,
    embedding_table: jax.Array,
    position_table: jax.Array,
) -> jax.Array:
    """Token embedding + positional embedding, summed.

    Args:
        token_ids:       (batch, seq_len), integer token IDs.
        embedding_table: (vocab_size, d_model).
        position_table:  (max_seq_len, d_model).

    Returns:
        (batch, seq_len, d_model)
    """
    seq_len = token_ids.shape[1]
    tok_emb = embed_tokens(token_ids, embedding_table)
    pos_emb = embed_positions(seq_len, position_table)
    return tok_emb + pos_emb
