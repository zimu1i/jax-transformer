import random

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

class TranslationTokenizer:
    
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.char_to_id = {ch: i for i, ch in enumerate(chars, 3)}
        self.id_to_char = {i: ch for ch, i in self.char_to_id.items()}
        self.vocab_size = len(chars) + 3
        
    def encode(self, text: str) -> list[int]:
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_char[i] for i in ids if i > 2)

def pad_sequence(
    token_ids: list[int],
    max_len: int,
) -> list[int]:
    # appends <eos> and pads with <pad>
    token_ids += [2]
    token_ids = token_ids + [0] * (max_len - len(token_ids))
    
    return token_ids


def load_pairs(path: str, max_len: int, limit: int, seed: int = 0) -> list[tuple[str, str]]:
    """Load (spanish, english) pairs from a tab-separated Tatoeba/Anki file."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                continue
            english, spanish = fields[0].lower(), fields[1].lower()
            # room for <bos> and <eos> on the target side
            if len(spanish) > max_len - 2 or len(english) > max_len - 2:
                continue
            pairs.append((spanish, english))

    # the file is sorted by length, so shuffle before taking a subset
    random.Random(seed).shuffle(pairs)
    return pairs[:limit]


def create_translation_batches(
    pairs: list[tuple[str,str]],
    src_tokenizer: TranslationTokenizer,
    tgt_tokenizer: TranslationTokenizer,
    max_src_len: int,
    max_tgt_len: int,
    batch_size: int,
):
    src_batches = []
    tgt_batches = []
    
    for i in range(len(pairs)):
        src_ids = src_tokenizer.encode(pairs[i][0])
        tgt_ids = tgt_tokenizer.encode(pairs[i][1])
        src_batches.append(pad_sequence(src_ids, max_src_len))
        tgt_ids = [1] + tgt_ids
        tgt_batches.append(pad_sequence(tgt_ids, max_tgt_len))
        
    num_pairs = (len(src_batches) // batch_size) * batch_size
    src_batches = src_batches[:num_pairs]
    tgt_batches = tgt_batches[:num_pairs]

    src_array = jnp.array(src_batches).reshape(-1, batch_size, max_src_len)
    tgt_array = jnp.array(tgt_batches).reshape(-1, batch_size, max_tgt_len)
    
    return src_array, tgt_array
    
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
