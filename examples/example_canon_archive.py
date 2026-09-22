"""Example: Process a canon archive through the polyvocoder.

Reads canon cells from a JSON file and renders each through the polyvocoder,
saving outputs to ./outputs/.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polyvocoder.pipeline_v2 import run_pipeline_v2
from polyvocoder.save_audio import save_wav


def main():
    archive_path = Path(__file__).parent.parent / "examples" / "sample_archive.json"
    if not archive_path.exists():
        print(f"Archive not found at {archive_path}")
        print("Run example_canon_archive.py with a real archive, or use example_minimal.py")
        return

    archive = json.load(open(archive_path))
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)

    for cell in archive[:5]:  # first 5 cells
        print(f"\n=== Cell {cell['id']} ===")
        result = run_pipeline_v2(cell["lore"], n_samples=1)
        sample = result["decoded_samples"][0]

        print(f"  JEV features: {[f'{v:.2f}' for v in result['features']]}")
        print(f"  Text: {sample['text'][:80]}...")
        print(f"  Image pattern: {sample['image'].split(chr(10))[0]}")

        save_wav(sample["audio"], str(out_dir / f"{cell['id']}.wav"))
        print(f"  Saved {out_dir / f'{cell[\"id\"]}.wav'}")


if __name__ == "__main__":
    main()
