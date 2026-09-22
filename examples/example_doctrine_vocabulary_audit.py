"""Example: audit which outputs use doctrinal vocabulary.

The polyvocoder heads are pre-baked with vocabulary pools that include
canonical Quilt substrate walker terms. This example audits how many
of those terms appear in each output.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polyvocoder.pipeline_v2 import run_pipeline_v2


DOCTRINE_TERMS = [
    "witness log", "canon gate", "oracle", "substrate",
    "FNV-1a", "scars", "cells", "witness",
    "amplitudes", "frequencies", "chord",
    "neon", "rain", "concrete", "memory",
]


def main():
    lore = "A story about a city that has cells and a gate."
    result = run_pipeline_v2(lore, n_samples=10)

    print(f"=== DOCTRINAL VOCABULARY AUDIT ===\n")
    print(f"Input lore: {lore}\n")
    print(f"Doctrine terms: {len(DOCTRINE_TERMS)}\n")

    for i, sample in enumerate(result["decoded_samples"]):
        text = sample["text"].lower()
        matches = [t for t in DOCTRINE_TERMS if t in text]

        print(f"Sample {i}: {len(matches)} terms matched")
        print(f"  Text: {sample['text']}")
        print(f"  Terms: {matches}")
        print()


if __name__ == "__main__":
    main()
