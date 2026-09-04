import jax
import jax.numpy as jnp


class CharTokenizer:
    """Character-level tokenizer. Maps each unique character to an integer."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_char[i] for i in ids)


def create_batches(
    token_ids: list[int],
    seq_len: int,
    batch_size: int,
) -> jax.Array:
    """Chunk a flat list of token IDs into fixed-size training batches.

    Each example is seq_len+1 tokens: the first seq_len are input,
    and tokens shifted by 1 are targets (handled in the training loop).

    Args:
        token_ids: Flat list of integer token IDs.
        seq_len:   Number of input tokens per example.
        batch_size: Examples per batch.

    Returns:
        (num_batches, batch_size, seq_len + 1) integer array.
        Leftover tokens that don't fill a complete batch are dropped.
    """
    example_len = seq_len + 1
    num_examples = len(token_ids) // example_len
    num_examples = (num_examples // batch_size) * batch_size

    usable = token_ids[: num_examples * example_len]
    arr = jnp.array(usable, dtype=jnp.int32).reshape(num_examples, example_len)
    return arr.reshape(-1, batch_size, example_len)
