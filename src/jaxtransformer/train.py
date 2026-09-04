import jax
import jax.numpy as jnp
import optax

from jaxtransformer.model import DecoderTransformer


def cross_entropy_loss(logits: jax.Array, targets: jax.Array) -> jax.Array:
    """Mean cross-entropy loss over all positions.

    Args:
        logits:  (batch, seq_len, vocab_size)
        targets: (batch, seq_len), integer token IDs.

    Returns:
        Scalar loss.
    """
    vocab_size = logits.shape[-1]
    one_hot = jax.nn.one_hot(targets, vocab_size)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


def make_train_step(model: DecoderTransformer, optimizer: optax.GradientTransformation):
    """Build a jitted train step function.

    Returns a function: (params, opt_state, batch) -> (params, opt_state, loss)
    """

    @jax.jit
    def train_step(params, opt_state, batch):
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        def loss_fn(params):
            logits = model.apply(params, inputs)
            return cross_entropy_loss(logits, targets)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss

    return train_step


def train(
    text: str,
    num_epochs: int = 3,
    seq_len: int = 64,
    batch_size: int = 8,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    d_ff: int = 256,
    lr: float = 3e-4,
    seed: int = 0,
    log_every: int = 50,
):
    """Train a small character-level Transformer on the given text.

    Returns the trained params, model, and tokenizer.
    """
    from jaxtransformer.data import CharTokenizer, create_batches

    tokenizer = CharTokenizer(text)
    token_ids = tokenizer.encode(text)
    batches = create_batches(token_ids, seq_len=seq_len, batch_size=batch_size)
    num_batches = batches.shape[0]

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Tokens: {len(token_ids):,}")
    print(f"Batches per epoch: {num_batches}")
    print()

    model = DecoderTransformer(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=seq_len,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
    )

    key = jax.random.PRNGKey(seed)
    dummy_input = batches[0][:, :-1]
    params = model.init(key, dummy_input)

    param_count = sum(p.size for p in jax.tree.leaves(params))
    print(f"Parameters: {param_count:,}")
    print()

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    train_step = make_train_step(model, optimizer)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for step in range(num_batches):
            params, opt_state, loss = train_step(params, opt_state, batches[step])
            epoch_loss += float(loss)

            global_step = epoch * num_batches + step
            if global_step % log_every == 0:
                print(f"epoch {epoch+1}/{num_epochs}  step {step}/{num_batches}  loss={float(loss):.4f}")

        avg_loss = epoch_loss / num_batches
        print(f"epoch {epoch+1}/{num_epochs}  avg_loss={avg_loss:.4f}")
        print()

    return params, model, tokenizer


def generate(
    model: DecoderTransformer,
    params,
    tokenizer,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    seed: int = 0,
) -> str:
    """Autoregressive text generation from a prompt."""
    token_ids = tokenizer.encode(prompt)
    key = jax.random.PRNGKey(seed)

    for _ in range(max_tokens):
        x = jnp.array([token_ids[-model.max_seq_len:]])
        logits = model.apply(params, x)
        next_logits = logits[0, -1] / temperature

        key, subkey = jax.random.split(key)
        next_id = int(jax.random.categorical(subkey, next_logits))
        token_ids.append(next_id)

    return tokenizer.decode(token_ids)
