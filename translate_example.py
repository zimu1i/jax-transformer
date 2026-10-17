"""Train the encoder-decoder Transformer on Spanish-English pairs, then translate.

Usage:
    python translate_example.py                 # train, save, then translate interactively
    python translate_example.py --load          # skip training, load the saved model
    python translate_example.py --epochs 20 --limit 40000

Expects data/spa.txt from https://www.manythings.org/anki/ (spa-eng.zip).
"""
import argparse
import pickle

import jax

from jaxtransformer.data import load_pairs
from jaxtransformer.train import train_translation, translate

CHECKPOINT = "data/translator.pkl"
NUM_HELD_OUT = 200


def unknown_chars(sentence, tokenizer):
    return sorted(set(sentence) - set(tokenizer.char_to_id))


def main():
    parser = argparse.ArgumentParser(description="Spanish to English translation")
    parser.add_argument("--data", default="data/spa.txt")
    parser.add_argument("--load", action="store_true", help="load the saved model instead of training")
    parser.add_argument("--limit", type=int, default=20000)
    parser.add_argument("--max-len", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    if args.load:
        with open(CHECKPOINT, "rb") as f:
            saved = pickle.load(f)
        params, model = saved["params"], saved["model"]
        src_tok, tgt_tok, held_out = saved["src_tok"], saved["tgt_tok"], saved["held_out"]
        max_len = saved["max_len"]
        print(f"Loaded model from {CHECKPOINT}\n")
    else:
        max_len = args.max_len
        pairs = load_pairs(args.data, max_len=max_len, limit=args.limit + NUM_HELD_OUT)
        train_pairs, held_out = pairs[:-NUM_HELD_OUT], pairs[-NUM_HELD_OUT:]
        print(f"Training on {len(train_pairs):,} pairs, holding out {len(held_out)}\n")

        params, model, src_tok, tgt_tok = train_translation(
            train_pairs,
            num_epochs=args.epochs,
            max_src_len=max_len,
            max_tgt_len=max_len,
            batch_size=args.batch_size,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            d_ff=args.d_ff,
            lr=args.lr,
            log_every=100,
        )
        with open(CHECKPOINT, "wb") as f:
            pickle.dump(dict(params=jax.device_get(params), model=model, src_tok=src_tok,
                             tgt_tok=tgt_tok, held_out=held_out, max_len=max_len), f)
        print(f"Saved model to {CHECKPOINT}\n")

    print("--- Sentences the model never saw during training ---")
    for spanish, english in held_out[:8]:
        if unknown_chars(spanish, src_tok):
            continue
        guess = translate(model, params, src_tok, tgt_tok, spanish, max_len=max_len)
        print(f"  es:    {spanish}\n  model: {guess}\n  real:  {english}\n")

    print(f"Type a Spanish sentence (up to {max_len - 1} characters), or press Enter to quit.")
    while True:
        try:
            sentence = input("es> ").strip().lower()
        except EOFError:
            break
        if not sentence:
            break
        if len(sentence) > max_len - 1:
            print(f"  Too long: the model was trained on sentences up to {max_len - 1} characters.")
            continue
        missing = unknown_chars(sentence, src_tok)
        if missing:
            print(f"  The model never saw these characters in training: {' '.join(missing)}")
            continue
        print("en>", translate(model, params, src_tok, tgt_tok, sentence, max_len=max_len))


if __name__ == "__main__":
    main()
