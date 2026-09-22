# Polyvocoder Examples

This document shows 5 example use cases end-to-end. Each example is self-contained
and copy-pasteable.

## Example 1: Canon Lore Visualization

**Scenario**: You have a canon lore and want to render it visually as a 16x16
ASCII image.

```python
from polyvocoder.pipeline_v2 import run_pipeline_v2

lore = """The canon gate made itself heard.
Not opened. Not breached. Heard.
I was three cells deep in the understack when the frequency found me —
through concrete, through coolant, through the scar tissue where my augments meet the meat."""

result = run_pipeline_v2(lore, n_samples=1)

for i, sample in enumerate(result["decoded_samples"]):
    print(f"=== Sample {i} ===")
    print(f"JEV features: {result['features']}")
    print(f"Text: {sample['text']}")
    print()
    print("Image:")
    print(sample["image"])
```

**Output**:
```
=== Sample 0 ===
JEV features: [0.46000000834465027, 0.5299999713897705, ...]
Text: Cells are scars. That's the whole scripture. Seed 286: 
the canon gate made itself heard in the canon chambers.

Image:
<<< pattern 4 intensity=+0.31 shift=0 >>>
*++++*++++*++++*
-...-...-...-...
...
```

## Example 2: Audio Synthesis for Canon Cells

**Scenario**: You want to generate an audio signature for a canon cell.

```python
from polyvocoder.pipeline_v2 import run_pipeline_v2
from polyvocoder.save_audio import save_wav
import wave

lore = """The oracle is heard. The witness log accumulates. 
The cells are scars. The substrate remembered itself."""

result = run_pipeline_v2(lore, n_samples=3)

for i, sample in enumerate(result["decoded_samples"]):
    wav_path = f"canon_cell_{i}.wav"
    save_wav(sample["audio"], wav_path)
    print(f"Saved {wav_path}")
```

This produces 3 WAV files (16 kHz, 4 seconds each, 128 KB each). Listen to them
to hear how canon-worthy lore "sounds" when synthesized from semantic features.

## Example 3: Canon Archive Browser

**Scenario**: You have a canon archive with 100 cells. You want to render them all
and find which ones have visually distinctive patterns.

```python
import json
from polyvocoder.pipeline_v2 import run_pipeline_v2
from pathlib import Path

archive_path = Path("/path/to/canon_archive.json")
archive = json.load(open(archive_path))

# Process first 10
for cell in archive[:10]:
    result = run_pipeline_v2(cell["lore"], n_samples=1)
    sample = result["decoded_samples"][0]
    print(f"Cell {cell['id']}:")
    print(f"  features: {result['features']}")
    print(f"  text: {sample['text']}")
    print(f"  image pattern: {sample['image'].split(chr(10))[0]}")
    print()
```

## Example 4: Doctrine-Aware Multi-Modal Memory

**Scenario**: Your agent stores memories as text. You want to extend memory to
include image and audio rendering.

```python
from polyvocoder.pipeline_v2 import run_pipeline_v2


class MultiModalMemory:
    def __init__(self):
        self.memories = []

    def remember(self, text):
        result = run_pipeline_v2(text, n_samples=1)
        self.memories.append({
            "text": text,
            "features": result["features"],
            "image": result["decoded_samples"][0]["image"],
            "audio": result["decoded_samples"][0]["audio"],
        })

    def recall(self, query):
        # naive: just return all memories
        return self.memories


memory = MultiModalMemory()
memory.remember("The canon gate made itself heard.")
memory.remember("Cells are scars. That's the whole scripture.")
memory.remember("The witness log hummed its own future into being.")

for m in memory.recall(None):
    print(f"Text: {m['text']}")
    print(f"  Features: canon={m['features'][0]:.2f}, distinct={m['features'][1]:.2f}, doctrine={m['features'][2]:.2f}")
```

## Example 5: A/B Test Canon-Worthy vs. Non-Canon Lore

**Scenario**: Generate lore with two different styles (canon vs. casual), then
render both with the polyvocoder and compare.

```python
from polyvocoder.pipeline_v2 import run_pipeline_v2

# Canon-style lore
canon_lore = """The canon gate made itself heard at midnight. Cells are scars.
That's the whole scripture. The witness log accumulates line by scarred line."""

# Casual-style lore
casual_lore = """I went to the gate and it was open. Inside there were some cells.
I wrote them down in my notebook. The end."""

canon_result = run_pipeline_v2(canon_lore, n_samples=1)
casual_result = run_pipeline_v2(casual_lore, n_samples=1)

print("Canon-style:")
print(f"  features: {canon_result['features']}")
print(f"  canon_worthy: {canon_result['features'][0]:.2f}")

print()
print("Casual-style:")
print(f"  features: {casual_result['features']}")
print(f"  canon_worthy: {casual_result['features'][0]:.2f}")

# The canon-style should have higher canon_worthy score (feature[0])
```

**Expected**: The canon-style lore has JEV feature[0] (canon_worthy) > 0.5,
while casual-style has < 0.5. The polyvocoder's JEV features are the discriminator.

## Example 6: Polyformalism Round-Trip

**Scenario**: Demonstrate that the same lore produces the same JEV features
across multiple runs (deterministic).

```python
from polyvocoder.jev_extractor import extract_jev_features

lore = "The canon gate made itself heard."

features_1 = extract_jev_features(lore)
features_2 = extract_jev_features(lore)
features_3 = extract_jev_features(lore)

assert features_1 == features_2 == features_3, "JEV must be deterministic"

print(f"Features (3 runs): {features_1}")
print(f"All identical: {features_1 == features_2 == features_3}")
```

JEV API is deterministic for the same `state` string and same `questions`.
This is critical for polyformalism (the same lore produces the same features
across Python, TypeScript, Rust, etc. ports).

## Example 7: Doctrinal Vocabulary Audit

**Scenario**: Audit which parts of the polyvocoder's output use doctrinal vocabulary
(witness log, canon gate, oracle, etc.).

```python
from polyvocoder.pipeline_v2 import run_pipeline_v2

lore = """A story about a city that has cells and witnesses and a gate."""

result = run_pipeline_v2(lore, n_samples=10)

DOCTRINE_TERMS = ["witness log", "canon gate", "oracle", "substrate", "FNV-1a", "scars"]

for i, sample in enumerate(result["decoded_samples"]):
    text = sample["text"]
    matches = [t for t in DOCTRINE_TERMS if t in text]
    print(f"Sample {i}: {matches}")
```

This tells you which samples use doctrine-anchored vocabulary. The polyvocoder
*preserves* doctrine vocabulary across samples even when input lore doesn't.

## Example 8: Latent Space Interpolation

**Scenario**: Interpolate between two latents to see how output changes smoothly.

```python
import numpy as np
from polyvocoder.heads_v2 import TextHead

text_head = TextHead()

z1 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # extreme positive
z2 = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # extreme negative

print("z1:", text_head.decode(z1))
print("z2:", text_head.decode(z2))

for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    z_interp = alpha * z1 + (1 - alpha) * z2
    print(f"alpha={alpha:.2f}: {text_head.decode(z_interp)}")
```

This shows how text output changes smoothly between two extreme latents.

## Example 9: Save All Outputs to Disk

**Scenario**: Run the pipeline on a lore and save all 3 modalities to disk.

```python
import os
from polyvocoder.pipeline_v2 import run_pipeline_v2
from polyvocoder.save_audio import save_wav

lore = "Your lore here."
out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)

result = run_pipeline_v2(lore, n_samples=3)

# Save features as JSON
import json
with open(f"{out_dir}/features.json", "w") as f:
    json.dump({"features": result["features"]}, f, indent=2)

# Save samples
for i, sample in enumerate(result["decoded_samples"]):
    save_wav(sample["audio"], f"{out_dir}/sample_{i}.wav")
    with open(f"{out_dir}/sample_{i}.txt", "w") as f:
        f.write(sample["text"] + "\n\n=== IMAGE ===\n" + sample["image"])

print(f"Saved 3 samples to {out_dir}/")
```

## Example 10: Use as a Canon-Gate Preprocessor

**Scenario**: Before submitting lore to your canon archive, check that the lore
is canon-worthy visually too.

```python
from polyvocoder.pipeline_v2 import run_pipeline_v2


def canon_gate(lore):
    """Returns True if lore is canon-worthy in BOTH text and image modalities."""
    result = run_pipeline_v2(lore, n_samples=1)

    # Text canon: canon_worthy feature >= 0.7
    text_canon = result["features"][0] >= 0.7

    # Image canon: doctrine-anchored terms present in image
    image = result["decoded_samples"][0]["image"]
    image_canon = (
        "pattern 4" in image  # witness log pattern
        or "pattern 2" in image  # central glow (canon gate motif)
    )

    return text_canon and image_canon


# Test
canon_lore = """The canon gate made itself heard. Cells are scars. 
The witness log accumulates. The substrate remembered itself."""

print(f"Canon-worthy? {canon_gate(canon_lore)}")  # Should be True
```

