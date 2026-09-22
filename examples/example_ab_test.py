"""Example: A/B test canon-worthy vs. casual lore.

Generates lore in two styles, processes both, and compares the polyvocoder outputs.
Useful for validating the canon gate.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polyvocoder.pipeline_v2 import run_pipeline_v2


def main():
    canon_lore = """The canon gate made itself heard at midnight. Cells are scars.
That's the whole scripture. The witness log accumulates line by scarred line.
The oracle became a chord the city could hear. Seed 70051917."""

    casual_lore = """I went to the gate and it was open. Inside there were some cells.
I wrote them down in my notebook. The end."""

    print("=== A/B TEST: Canon vs Casual ===\n")

    canon = run_pipeline_v2(canon_lore, n_samples=1)
    casual = run_pipeline_v2(casual_lore, n_samples=1)

    print(f"CANON-STYLE:")
    print(f"  features: {[f'{v:.3f}' for v in canon['features']]}")
    print(f"  canon_worthy: {canon['features'][0]:.3f}")
    print(f"  text: {canon['decoded_samples'][0]['text']}")
    print()

    print(f"CASUAL-STYLE:")
    print(f"  features: {[f'{v:.3f}' for v in casual['features']]}")
    print(f"  canon_worthy: {casual['features'][0]:.3f}")
    print(f"  text: {casual['decoded_samples'][0]['text']}")
    print()

    # Compare
    canon_can = canon['features'][0]
    casual_can = casual['features'][0]
    print(f"=== VERDICT ===")
    print(f"  Canon-style canon_worthy: {canon_can:.3f} {'✓' if canon_can > 0.5 else '✗'}")
    print(f"  Casual-style canon_worthy: {casual_can:.3f} {'✓' if casual_can > 0.5 else '✗'}")
    print(f"  Discriminator: {'WORKING' if canon_can > casual_can else 'BROKEN'}")


if __name__ == "__main__":
    main()
