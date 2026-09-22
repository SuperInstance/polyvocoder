"""Example: minimal — process one lore through the polyvocoder.

This is the simplest possible use case. Perfect for testing your install.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polyvocoder.pipeline_v2 import run_pipeline_v2


def main():
    lore = "The canon gate made itself heard."

    print(f"Input lore: {lore}\n")
    result = run_pipeline_v2(lore, n_samples=1)

    print(f"JEV features: {[f'{v:.3f}' for v in result['features']]}")
    print()

    sample = result["decoded_samples"][0]
    print(f"Text: {sample['text']}\n")
    print("Image:")
    print(sample["image"])
    print()

    audio = sample["audio"]
    print(f"Audio: {audio.shape[0]} samples, range [{audio.min():.3f}, {audio.max():.3f}]")


if __name__ == "__main__":
    main()
