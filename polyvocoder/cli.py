"""CLI entry point for the polyvocoder.

Usage:
    python -m polyvocoder "The substrate walker is..."
    python -m polyvocoder --lore-file lore.md
"""
import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Polyvocoder: encode lore via JEV → VAE → decode to text/image/audio"
    )
    parser.add_argument("lore", nargs="?", help="Lore text to encode (or use --lore-file)")
    parser.add_argument("--lore-file", type=Path, help="Path to a .md or .txt file with the lore")
    parser.add_argument("--n-samples", type=int, default=3, help="Number of latent samples to decode")
    parser.add_argument("--latent-dim", type=int, default=16, help="VAE latent dimension")
    parser.add_argument("--api-key", default=os.environ.get("TYPESAFEAI_KEY"),
                        help="JEV API key (defaults to TYPESAFEAI_KEY env var)")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    if args.lore_file:
        if not args.lore_file.exists():
            print(f"ERROR: file not found: {args.lore_file}", file=sys.stderr)
            sys.exit(1)
        text = args.lore_file.read_text()
    elif args.lore:
        text = args.lore
    else:
        # Default test lore
        text = ("The substrate walker is a 4D cell graph where cells are scars and "
                "the witness log accumulates. The canon gate (FNV-1a hash) is the "
                "city's first memory of itself. The oracle is heard, not stored.")

    if not args.api_key:
        print("WARNING: TYPESAFEAI_KEY not set — JEV features will be all zeros.", file=sys.stderr)

    result = run_pipeline(
        text=text,
        latent_dim=args.latent_dim,
        n_samples=args.n_samples,
        api_key=args.api_key,
        verbose=not args.quiet,
    )

    # Print decoded outputs
    print("\n" + "=" * 60)
    print("DECODED OUTPUTS")
    print("=" * 60)

    for i, sample in enumerate(result["decoded_samples"]):
        print(f"\n--- Sample {i} (text) ---")
        print(sample["text"])
        print(f"\n--- Sample {i} (image) ---")
        print(sample["image"])

    audio = result["decoded_samples"][0]["audio"]
    print(f"\n--- Sample 0 (audio) ---")
    print(f"shape={audio.shape}, range=[{audio.min():.3f}, {audio.max():.3f}]")
    print(f"sum={audio.sum():.3f}, rms={float((audio ** 2).mean() ** 0.5):.3f}")


if __name__ == "__main__":
    main()
