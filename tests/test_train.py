import jax
import jax.numpy as jnp
import optax

from jaxtransformer.model import DecoderTransformer
from jaxtransformer.train import cross_entropy_loss, make_train_step


def test_cross_entropy_loss_range():
    """Loss must be positive and bounded. For random logits over vocab_size=V,
    expected loss is roughly ln(V)."""
    key = jax.random.PRNGKey(0)
    logits = jax.random.normal(key, (2, 8, 32))
    targets = jax.random.randint(jax.random.PRNGKey(1), (2, 8), 0, 32)

    loss = cross_entropy_loss(logits, targets)
    assert loss > 0, "Loss should be positive"
    assert loss < 10, f"Loss unexpectedly high: {loss}"


def test_cross_entropy_perfect_prediction():
    """When logits strongly favor the correct token, loss should be near zero."""
    targets = jnp.array([[0, 1, 2]])
    logits = jnp.zeros((1, 3, 4))
    logits = logits.at[0, 0, 0].set(100.0)
    logits = logits.at[0, 1, 1].set(100.0)
    logits = logits.at[0, 2, 2].set(100.0)

    loss = cross_entropy_loss(logits, targets)
    assert loss < 1e-3, f"Loss should be ~0 for perfect predictions, got {loss}"


def test_train_step_reduces_loss():
    """A single training step must reduce the loss. If it doesn't, either
    the gradient computation or the optimizer update is broken."""
    model = DecoderTransformer(
        vocab_size=16, max_seq_len=8, d_model=16,
        num_heads=2, num_layers=1, d_ff=64,
    )
    key = jax.random.PRNGKey(0)
    batch = jax.random.randint(jax.random.PRNGKey(1), (4, 9), 0, 16)  # seq_len+1=9

    params = model.init(key, batch[:, :-1])
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    train_step = make_train_step(model, optimizer)

    _, _, loss_before = train_step(params, opt_state, batch)

    # Run a few steps
    for _ in range(5):
        params, opt_state, loss = train_step(params, opt_state, batch)

    assert loss < loss_before, (
        f"Loss did not decrease: {loss_before:.4f} -> {loss:.4f}"
    )
