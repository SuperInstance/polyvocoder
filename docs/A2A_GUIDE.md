# Agent-to-Agent Guide: How to Use the Polyvocoder from Your Agent

**Target audience**: another LLM-based agent (or an agent-builder) who wants to
integrate canon-aware multi-modal generation into their workflow.

## TL;DR (60-second integration)

```python
import sys
sys.path.insert(0, "/path/to/polyvocoder")

from polyvocoder.pipeline_v2 import run_pipeline_v2

result = run_pipeline_v2(
    lore_text="Your canon lore here.",
    n_samples=3,
    vae_seed=42,
)

# result["features"] is the 6-dim JEV vector
# result["decoded_samples"][i] is one sample:
#   - sample["text"]:  doctrine-anchored prose
#   - sample["image"]: 16x16 ASCII art
#   - sample["audio"]: 4-second 16kHz waveform (np.float32)
```

## What This Actually Does

For ANY input text (your lore, your agent's output, your user's prompt), the
polyvocoder will:

1. **Ask JEV 6 questions** about your text. JEV returns calibrated probabilities
   for canon-worthiness, distinct voice, doctrine-anchor, voice-fit, novelty,
   and density. Cost: ~$0.04/1M input tokens (about 100 lores per cent).

2. **Run a tiny VAE** to compress those 6 features into a 6-dim latent z. The
   VAE is a single file (`polyvocoder/vae.py`), no torch required. Training
   is ~30 epochs of Adam, takes <1 second.

3. **Decode z via 3 heads**:
   - TextHead: doctrine-anchored prose templates (witness log, canon gate, oracle,
     substrate, FNV-1a, scars, cells)
   - ImageHead: 16x16 ASCII art (6 base patterns, intensity modulation)
   - AudioHead: 4-second 16kHz waveform (harmonic synthesis, 4 envelope shapes)

4. **Return** all three modalities for n_samples different latents.

## When to Use It

Use the polyvocoder when you have:
- A canon-like lore (cyberpunk-noir, poetic, image-rich) and want to test its
  multi-modal canon-ness.
- A need to render the same semantic content in text + image + audio form
  (e.g., for a canon archive, a museum exhibit, a multi-modal agent).
- A workflow where JEV canon-gating is already part of the loop, and you want
  to extend the gate to sensory modalities.

Do NOT use the polyvocoder for:
- Pixel-perfect image generation (use SDXL-Lightning or similar).
- Speech synthesis (use ElevenLabs or similar).
- Production multi-modal output (the heads are templates, not learned models).

## Architectural Patterns You Can Steal

### Pattern 1: JEV as Semantic Encoder

Replace any "extract features from text" step with a JEV probe:

```python
import requests
TYPESAFEAI_KEY = "your-jev-key"

def extract_jev_features(text):
    state = f"My agent's output: {text}"
    questions = {
        "feature_1": {"type": "noul", "instructions": "Your first feature question?"},
        "feature_2": {"type": "noul", "instructions": "Your second feature question?"},
        # ...
    }
    resp = requests.post(
        "https://api.typesafe.ai/v1/systemone",
        headers={"Authorization": f"Bearer {TYPESAFEAI_KEY}"},
        json={"state": state, "questions": questions},
    ).json()
    return [resp["answers"][k]["noul"] for k in questions]
```

Then use the 6-dim feature vector as the input to ANY downstream pipeline (VAE,
linear classifier, similarity search, etc.).

### Pattern 2: Universal Head Pattern (BigVGAN-style)

```python
class Head:
    def decode(self, z):
        raise NotImplementedError

class MyTextHead(Head):
    def decode(self, z):
        # pick template by argmax of z
        template_idx = int(np.argmax(np.abs(z))) % len(self.TEMPLATES)
        template = self.TEMPLATES[template_idx]
        # fill slots with words from pools keyed by z
        return template.format(**{slot: pool[hash(z.tobytes() + slot.encode()) % len(pool)]
                                   for slot, pool in self.POOLS.items()})

class MyAudioHead(Head):
    def decode(self, z):
        # use z coordinates as synthesis parameters
        f0 = 200 + abs(z[0]) * 600  # fundamental frequency
        n_harm = int(abs(z[1]) * 8) + 2  # number of harmonics
        # synthesize
        return synthesize_waveform(f0, n_harm, duration=4.0)

# Same z, multiple heads, multiple modalities
```

### Pattern 3: Polyformalism via Schema Parity

The Python and TypeScript bindings share an identical JSON schema:

```python
# Python
@dataclass
class JEVFeatures:
    canonWorthy: float
    distinctVoice: float
    doctrineAnchor: float
    voiceFit: float
    novelty: float
    density: float
```

```typescript
// TypeScript
interface JEVFeatures {
  canonWorthy: number;
  distinctVoice: number;
  doctrineAnchor: number;
  voiceFit: number;
  novelty: number;
  density: number;
}
```

Same fields, same types, same canonical meaning. This is the polyformalism
contract: the schema is the interface; implementations can vary.

### Pattern 4: Tiny VAE Without torch

The VAE in `polyvocoder/vae.py` is ~150 lines of numpy. It uses:
- Adam optimization (manually implemented)
- Numerical gradients via finite differences (10-30 dimensions per parameter)
- Clipping for numerical stability

This pattern works for any tiny VAE (<1M parameters). For larger VAEs, use torch.

## Common Integration Scenarios

### Scenario A: Add Canon-Checking to Your Lore Generator

```python
# Before: generate lore
lore = my_lore_generator(prompt)

# After: generate lore + verify canon
lore = my_lore_generator(prompt)
features = extract_jev_features(lore)
is_canon = features[0] >= 0.7  # canon_worthy threshold
if is_canon:
    multi_modal = run_pipeline_v2(lore)
    save_canon_cell(lore, multi_modal)
```

### Scenario B: Multi-Modal Memory in Your Agent

```python
# When your agent saves a memory
memory = your_agent.think(context)
result = run_pipeline_v2(memory)
your_agent.memory_store.add({
    "text": memory,
    "image": result["decoded_samples"][0]["image"],
    "audio": result["decoded_samples"][0]["audio"],
    "features": result["features"],
})
```

### Scenario C: Browse a Canon Archive Visually

```python
for cell in load_canon_archive():
    result = run_pipeline_v2(cell.lore)
    print(f"Cell {cell.id}:")
    print(result["decoded_samples"][0]["image"])
    # If image looks "canon" (e.g., contains recognizable motif), it's canon-anchored
    # If image is generic noise, the lore may be canon in text only
```

## Things You Can Customize

### Add a new question to JEV feature extraction

Edit `polyvocoder/jev_extractor.py` → add a question to the `questions` dict.
This extends the feature vector to 7 dimensions. Update VAE `input_dim` to 7.

### Add a new modality head

Create `polyvocoder/heads/your_head.py`:

```python
from polyvocoder.heads_v2 import Head
import numpy as np

class MyVideoHead(Head):
    def decode(self, z):
        # 30 frames at 8x8 ASCII, animated by z
        frames = []
        for t in range(30):
            frame = generate_frame(z, t)
            frames.append(frame)
        return frames
```

Register in `polyvocoder/pipeline_v2.py`.

### Add a new language binding

For Swift, Go, Rust, Kotlin, etc., follow the pattern in
`SuperInstance/polyvocoder-bindings` (TypeScript):

1. Define a class with the same interface as `PolyvocoderClient`
2. Implement HTTP client to a deployed Polyvocoder service
3. Mirror the Python dataclass types in your language
4. Document with examples

## What You Get vs. What You Don't

| You Get | You Don't Get |
|---|---|
| Canon-aware prose rendering | Pixel-perfect images |
| 6 base ASCII art patterns | Arbitrary-resolution images |
| 4-second synthetic audio | Speech synthesis or music |
| 100% reproducible output | Continuous variation |
| No GPU dependency | Deep learning fidelity |
| $0.04/1M tokens cost | Free local generation |

## Fleet Canary

The polyvocoder pins to the Quilt substrate walker fleet canary:
**`fnv1a-64('café Δ 日本語') = 0x024a555471370b18d`**

This is the canary hash that all 6 polyformalism ports (Python, TypeScript, Rust,
Bash, JS ESM, C#/.NET) compute byte-exact. If your integration produces a different
hash, your environment has a unicode or FNV-1a implementation difference.

To verify the canary:
```python
from polyvocoder.canary import canary_hash
assert canary_hash() == 0x024a555471370b18d
```

## Failure Modes

### "JEV returns errors"

JEV API has rate limits (~5 QPS for the Lite tier). Batch your probes.

### "VAE produces NaN losses"

The numerical gradient is sensitive to epsilon. Use `eps=1e-3` (default). For
smaller VAEs (<50 dims), `eps=1e-2` may work.

### "All text outputs look the same"

The latent z is dominated by VAE mean (μ) rather than reparameterized noise (σ).
Add more noise or use a smaller KL weight (try `kl_weight=0.001`).

### "Audio outputs sound the same"

The AudioHead uses z[0:5] for synthesis parameters. If z is dominated by mean,
all samples produce similar sounds. Sample from `vae.reparameterize(mu, log_sigma)`
multiple times with different epsilon.

## License

MIT — Casey / SuperInstance, Sept 22, 2026

