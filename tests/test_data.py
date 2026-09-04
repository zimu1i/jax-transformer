import jax.numpy as jnp

from jaxtransformer.data import CharTokenizer, create_batches


def test_tokenizer_roundtrip():
    """Encoding then decoding must return the original text."""
    text = "hello world 123!?"
    tok = CharTokenizer(text)
    assert tok.decode(tok.encode(text)) == text


def test_tokenizer_vocab_size():
    """Vocab size must equal the number of unique characters."""
    text = "aabbbcccc"
    tok = CharTokenizer(text)
    assert tok.vocab_size == 3


def test_create_batches_shape():
    """Batches must have shape (num_batches, batch_size, seq_len + 1).
    The +1 is for the target: input is [:seq_len], target is [1:seq_len+1]."""
    ids = list(range(200))
    batches = create_batches(ids, seq_len=8, batch_size=4)

    assert batches.ndim == 3
    assert batches.shape[1] == 4
    assert batches.shape[2] == 9  # seq_len + 1


def test_create_batches_content():
    """Batch content must match the original token IDs in order."""
    ids = list(range(100))
    batches = create_batches(ids, seq_len=4, batch_size=2)

    flat = batches.reshape(-1)
    expected_len = len(flat)
    assert jnp.array_equal(flat, jnp.array(ids[:expected_len]))
