"""Train a small character-level Transformer on a text file.

Usage:
    python train_example.py data.txt
    python train_example.py data.txt --epochs 10 --d-model 128 --num-layers 4

If no file is given, trains on a small built-in example.
"""
import argparse
from jaxtransformer.train import train, generate


EXAMPLE_TEXT = (
    "the quick brown fox jumps over the lazy dog. "
    "a stitch in time saves nine. "
    "all that glitters is not gold. "
    "actions speak louder than words. "
) * 200


def main():
    parser = argparse.ArgumentParser(description="Train a character-level Transformer")
    parser.add_argument("file", nargs="?", help="Text file to train on")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--prompt", type=str, default="the ")
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            text = f.read()
    else:
        print("No input file given, using built-in example text.\n")
        text = EXAMPLE_TEXT

    params, model, tokenizer = train(
        text,
        num_epochs=args.epochs,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        d_ff=args.d_ff,
        lr=args.lr,
    )

    print("--- Generated text ---")
    output = generate(model, params, tokenizer, args.prompt, max_tokens=args.max_tokens)
    print(output)


if __name__ == "__main__":
    main()
