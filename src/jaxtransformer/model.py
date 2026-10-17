import jax
import jax.numpy as jnp
import flax.linen as nn

from jaxtransformer.attention import (
    multihead_causal_self_attention,
    multihead_self_attention,
    multihead_cross_attention,
    make_causal_mask,
    scaled_dot_product_attention,
)
from jaxtransformer.block import layer_norm, feed_forward


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention as a Flax module.

    Wraps our hand-built attention — the math inside is identical to
    multihead_causal_self_attention(), but Flax manages the weight matrices.
    """
    num_heads: int
    d_model: int

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        head_dim = self.d_model // self.num_heads
        init = nn.initializers.xavier_uniform()

        w_q = self.param("w_q", init, (self.d_model, self.d_model))
        w_k = self.param("w_k", init, (self.d_model, self.d_model))
        w_v = self.param("w_v", init, (self.d_model, self.d_model))
        w_o = self.param("w_o", init, (self.d_model, self.d_model))

        return multihead_causal_self_attention(
            x, w_q, w_k, w_v, w_o, self.num_heads,
        )


class FeedForward(nn.Module):
    """Position-wise feed-forward network as a Flax module."""
    d_model: int
    d_ff: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        init_w = nn.initializers.xavier_uniform()
        init_b = nn.initializers.zeros_init()

        w1 = self.param("w1", init_w, (self.d_model, self.d_ff))
        b1 = self.param("b1", init_b, (self.d_ff,))
        w2 = self.param("w2", init_w, (self.d_ff, self.d_model))
        b2 = self.param("b2", init_b, (self.d_model,))

        return feed_forward(x, w1, b1, w2, b2)


class TransformerBlock(nn.Module):
    """One pre-norm decoder block: attention + FFN, each with residual + layernorm."""
    num_heads: int
    d_model: int
    d_ff: int

    @nn.compact
    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        # --- Sub-layer 1: attention ---
        ln1_gamma = self.param("ln1_gamma", nn.initializers.ones_init(), (self.d_model,))
        ln1_beta  = self.param("ln1_beta",  nn.initializers.zeros_init(), (self.d_model,))
        normed = layer_norm(x, ln1_gamma, ln1_beta)
        attn_out, weights = CausalSelfAttention(
            num_heads=self.num_heads, d_model=self.d_model, name="attention",
        )(normed)
        x = x + attn_out

        # --- Sub-layer 2: feed-forward ---
        ln2_gamma = self.param("ln2_gamma", nn.initializers.ones_init(), (self.d_model,))
        ln2_beta  = self.param("ln2_beta",  nn.initializers.zeros_init(), (self.d_model,))
        normed = layer_norm(x, ln2_gamma, ln2_beta)
        ff_out = FeedForward(
            d_model=self.d_model, d_ff=self.d_ff, name="ff",
        )(normed)
        x = x + ff_out

        return x, weights
        
class SelfAttention(nn.Module):
    """Multi-head self-attention (no causal mask) as a Flax module."""
    num_heads: int
    d_model: int

    @nn.compact
    def __call__(self, x, mask=None):
        init = nn.initializers.xavier_uniform()
        w_q = self.param("w_q", init, (self.d_model, self.d_model))
        w_k = self.param("w_k", init, (self.d_model, self.d_model))
        w_v = self.param("w_v", init, (self.d_model, self.d_model))
        w_o = self.param("w_o", init, (self.d_model, self.d_model))
        return multihead_self_attention(x, w_q, w_k, w_v, w_o, self.num_heads, mask=mask)


class CrossAttention(nn.Module):
    """Multi-head cross-attention as a Flax module."""
    num_heads: int
    d_model: int

    @nn.compact
    def __call__(self, x_q, x_kv, mask=None):
        init = nn.initializers.xavier_uniform()
        w_q = self.param("w_q", init, (self.d_model, self.d_model))
        w_k = self.param("w_k", init, (self.d_model, self.d_model))
        w_v = self.param("w_v", init, (self.d_model, self.d_model))
        w_o = self.param("w_o", init, (self.d_model, self.d_model))
        return multihead_cross_attention(x_q, x_kv, w_q, w_k, w_v, w_o, self.num_heads, mask=mask)


class EncoderBlock(nn.Module):
    num_heads: int
    d_model: int
    d_ff: int

    @nn.compact
    def __call__(self, x, mask=None):
        # sublayer 1: self-attention + residual
        ln1_gamma = self.param("ln1_gamma", nn.initializers.ones_init(), (self.d_model,))
        ln1_beta  = self.param("ln1_beta",  nn.initializers.zeros_init(), (self.d_model,))
        normed = layer_norm(x, ln1_gamma, ln1_beta)
        attn_out, weights = SelfAttention(
            num_heads=self.num_heads, d_model=self.d_model, name="attention",
        )(normed, mask=mask)
        x = x + attn_out

        # sublayer 2: FFN + residual
        ln2_gamma = self.param("ln2_gamma", nn.initializers.ones_init(), (self.d_model,))
        ln2_beta  = self.param("ln2_beta",  nn.initializers.zeros_init(), (self.d_model,))
        normed = layer_norm(x, ln2_gamma, ln2_beta)
        ff_out = FeedForward(
            d_model=self.d_model, d_ff=self.d_ff, name="ff",
        )(normed)
        x = x + ff_out

        return x, weights


class DecoderBlock(nn.Module):
    num_heads: int
    d_model: int
    d_ff: int

    @nn.compact
    def __call__(self, x, encoder_out, src_mask=None):
        # sublayer 1: causal self-attention + residual
        ln1_gamma = self.param("ln1_gamma", nn.initializers.ones_init(), (self.d_model,))
        ln1_beta  = self.param("ln1_beta",  nn.initializers.zeros_init(), (self.d_model,))
        normed = layer_norm(x, ln1_gamma, ln1_beta)
        attn_out, _ = CausalSelfAttention(
            num_heads=self.num_heads, d_model=self.d_model, name="self_attention",
        )(normed)
        x = x + attn_out

        # sublayer 2: cross-attention + residual
        ln2_gamma = self.param("ln2_gamma", nn.initializers.ones_init(), (self.d_model,))
        ln2_beta  = self.param("ln2_beta",  nn.initializers.zeros_init(), (self.d_model,))
        normed = layer_norm(x, ln2_gamma, ln2_beta)
        cross_out, cross_weights = CrossAttention(
            num_heads=self.num_heads, d_model=self.d_model, name="cross_attention",
        )(normed, encoder_out, mask=src_mask)
        x = x + cross_out

        # sublayer 3: FFN + residual
        ln3_gamma = self.param("ln3_gamma", nn.initializers.ones_init(), (self.d_model,))
        ln3_beta  = self.param("ln3_beta",  nn.initializers.zeros_init(), (self.d_model,))
        normed = layer_norm(x, ln3_gamma, ln3_beta)
        ff_out = FeedForward(
            d_model=self.d_model, d_ff=self.d_ff, name="ff",
        )(normed)
        x = x + ff_out

        return x, cross_weights


        
class EncoderDecoderTransformer(nn.Module):
    src_vocab_size: int    # Spanish vocabulary size
    tgt_vocab_size: int    # English vocabulary size
    max_seq_len: int
    d_model: int
    num_heads: int
    num_layers: int
    d_ff: int
    pad_id: int = 0

    @nn.compact
    def __call__(self, src_ids: jax.Array, tgt_ids: jax.Array) -> jax.Array:
        
        # Step A: Embed source and target (separate tables)
        src_seq_len = src_ids.shape[1]
        tgt_seq_len = tgt_ids.shape[1]
        batch = src_ids.shape[0]
        
        # source and target token/position embeddings
        src_token_embedding = self.param("source token embedding", nn.initializers.normal(stddev=0.02),
                                         (self.src_vocab_size, self.d_model))
        src_position_embedding = self.param("source position embedding", nn.initializers.normal(stddev=0.02),
                                         (self.max_seq_len, self.d_model))
        tgt_token_embedding = self.param("target token embedding", nn.initializers.normal(stddev=0.02),
                                         (self.tgt_vocab_size, self.d_model))
        tgt_position_embedding = self.param("target position embedding",nn.initializers.normal(stddev=0.02), 
                                            (self.max_seq_len, self.d_model))

        # input embedding
        src_emb = src_token_embedding[src_ids] + src_position_embedding[:src_seq_len]
        tgt_emb = tgt_token_embedding[tgt_ids] + tgt_position_embedding[:tgt_seq_len]
        
        # Step B: Build source padding mask
        src_mask = (src_ids != self.pad_id) # shape (batch, src_seq_len)
        src_mask = src_mask.reshape(batch, 1, 1, src_seq_len)
        
        # Step C: Run encoder block
        x = src_emb
        for i in range(self.num_layers):
            x, _ = EncoderBlock(num_heads = self.num_heads, d_model = self.d_model, 
                                d_ff = self.d_ff, name=f"encoder_{i}") (x, 
                                                                        mask=src_mask)
        encoder_out = x
            
        # Step D: Run decoder blocks (passing encoder output)
        x = tgt_emb
        for i in range(self.num_layers):
            x, _ = DecoderBlock(num_heads = self.num_heads, d_model = self.d_model, d_ff = self.d_ff, name=f"decoder_{i}") (x, 
                                                                                                                            encoder_out, src_mask=src_mask)
        decoder_out = x
        # Step E: Final layernorm + project to target vocab logits
        ln_f_gamma = self.param("ln_f_gamma", nn.initializers.ones_init(), (self.d_model,))
        ln_f_beta = self.param("ln_f_beta", nn.initializers.zeros_init(), (self.d_model,))
        normed = layer_norm(decoder_out, ln_f_gamma, ln_f_beta)
        logits = normed @ tgt_token_embedding.T
        return logits
        
class DecoderTransformer(nn.Module):
    """Full decoder-only Transformer: embeddings → N blocks → layernorm → logits."""
    vocab_size: int
    max_seq_len: int
    d_model: int
    num_heads: int
    num_layers: int
    d_ff: int

    @nn.compact
    def __call__(self, token_ids: jax.Array) -> jax.Array:
        batch, seq_len = token_ids.shape

        # --- Embeddings ---
        tok_emb_table = self.param(
            "token_embedding",
            nn.initializers.normal(stddev=0.02),
            (self.vocab_size, self.d_model),
        )
        pos_emb_table = self.param(
            "position_embedding",
            nn.initializers.normal(stddev=0.02),
            (self.max_seq_len, self.d_model),
        )
        x = tok_emb_table[token_ids] + pos_emb_table[:seq_len]

        # --- Transformer blocks ---
        for i in range(self.num_layers):
            x, _ = TransformerBlock(
                num_heads=self.num_heads,
                d_model=self.d_model,
                d_ff=self.d_ff,
                name=f"block_{i}",
            )(x)

        # --- Final layer norm ---
        ln_f_gamma = self.param("ln_f_gamma", nn.initializers.ones_init(), (self.d_model,))
        ln_f_beta  = self.param("ln_f_beta",  nn.initializers.zeros_init(), (self.d_model,))
        x = layer_norm(x, ln_f_gamma, ln_f_beta)

        # --- Project to vocabulary logits ---
        # Weight tying: reuse the token embedding table as the output projection.
        # This is standard in GPT-2 and saves vocab_size * d_model parameters.
        logits = x @ tok_emb_table.T

        return logits
