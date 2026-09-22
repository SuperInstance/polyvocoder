# Polyvocoder Architecture

## High-Level

```
Input lore text
   │
   ▼
JEV (Typesafe.ai System One, POST /v1/systemone)
   │  6 calibrated probabilities
   ▼
6-dim feature vector
   │
   ▼
Tiny VAE (numpy-only)
   │  encode → reparameterize → decode
   ▼
6-dim latent z (per sample)
   │
   ▼
┌──────────────┬──────────────┬──────────────┐
│ TextHead     │ ImageHead    │ AudioHead    │
└──────────────┴──────────────┴──────────────┘
   │              │              │
   ▼              ▼              ▼
prose          16×16 ASCII    4-sec 16kHz
                              waveform
```

## Module Layout

```
polyvocoder/
├── __init__.py            # public API exports
├── jev_extractor.py       # JEV feature extraction (calls Typesafe.ai API)
├── vae.py                 # tiny VAE, numpy-only, Adam training
├── heads.py               # v1 heads (basic)
├── heads_v2.py            # v2 heads (richer — recommended)
├── pipeline.py            # v1 end-to-end
├── pipeline_v2.py         # v2 end-to-end (uses heads_v2)
├── save_audio.py          # WAV export for audio outputs
└── cli.py                 # `python -m polyvocoder` entry point
```

## Data Flow

### Step 1: JEV Feature Extraction

```python
# jev_extractor.py:extract_jev_features(text)
questions = {
    "canon_worthy":    {"type": "noul", "instructions": "Canon-worthy?"},
    "distinct_voice":  {"type": "noul", "instructions": "Distinct voice?"},
    "doctrine_anchor": {"type": "noul", "instructions": "Doctrine-anchored?"},
    "voice_fit":       {"type": "noul", "instructions": "Voice archetype fit?"},
    "novelty":         {"type": "noul", "instructions": "Novel imagery?"},
    "density":         {"type": "noul", "instructions": "Image density (1.0+)?"},
}

resp = requests.post(
    "https://api.typesafe.ai/v1/systemone",
    headers={"Authorization": f"Bearer {TYPESAFEAI_KEY}"},
    json={"state": text, "questions": questions},
)
features = [resp["answers"][k]["noul"] for k in questions]
```

**Cost**: ~$0.04/1M input tokens, output free. A 200-word lore is ~300 tokens,
so ~$0.012 per feature extraction.

**Time**: 2-5 seconds for a single probe.

**Determinism**: JEV results are repeatable for the same `state` string.

### Step 2: Tiny VAE

```python
# vae.py:VAE(input_dim=6, latent_dim=6, hidden_dim=16)
# Training: ~30 epochs Adam with numerical gradients
# Augmentation: 1 feature vector + 20 noisy copies (σ=0.05)

class VAE:
    def encode(self, x) -> (μ, log_σ):
        h = relu(x @ W_enc1 + b_enc1)
        μ = h @ W_mu + b_mu
        log_σ = clip(h @ W_log_sigma + b_log_sigma, -3, 3)
        return μ, log_σ

    def reparameterize(self, μ, log_σ):
        σ = exp(log_σ)
        ε ~ N(0, 1)
        return μ + σ * ε

    def decode(self, z):
        h = relu(z @ W_dec1 + b_dec1)
        return clip(h @ W_dec_out + b_dec_out, -2, 2)

    def loss(self, x, x̂, μ, log_σ, kl_weight=0.01):
        recon_loss = mean((x - x̂) ** 2)
        kl_loss = -0.5 * mean(1 + log_σ - μ ** 2 - exp(log_σ))
        return recon_loss + kl_weight * kl_loss
```

**Why 6-dim latent for 6-dim input?** The VAE learns a stochastic projection.
The reparameterization lets us sample diverse latents from a single input.
The KL term keeps the latent close to N(0, 1) so the heads see natural inputs.

**Adam optimizer (manually implemented)**:
```python
m[name] = β1 * m[name] + (1-β1) * g     # first moment
v[name] = β2 * v[name] + (1-β2) * g²    # second moment
m̂ = m[name] / (1 - β1^t)
v̂ = v[name] / (1 - β2^t)
update = lr * m̂ / (sqrt(v̂) + ε)
param -= update
```

### Step 3: Heads

Each head is a pure function `z → output`. The heads do NOT learn during the
pipeline; they are pre-baked but parametric (their behavior varies by z).

#### TextHead

```python
class TextHead:
    TEMPLATES = [
        "Witness log of seed {seed}: cells became {scar}; oracle spoke in {voice}; canon gate measured {measure}.",
        # ... 8 doctrine-anchored templates ...
    ]

    ADJ_POOL = ["neon-drenched", "fractal-bright", "quantum-pale", ...]
    NOUN_POOL = ["sprawl", "matrix", "grid", "lattice", ...]
    SCAR_POOL = ["amplitudes", "scars", "cells", "witness log entries", ...]
    # ... 14 word pools total ...

    def decode(self, z):
        template_idx = argmax(|z|) % len(TEMPLATES)
        seed = int(|z[0]| * 1000) % 999999
        # ... fill template slots from pools, weighted by z ...
```

#### ImageHead

```python
class ImageHead:
    PATTERNS = [
        # 6 base patterns as multi-line strings
        # 0: vertical stripes
        # 1: density gradient (left to right)
        # 2: central glow (diamond + X)
        # 3: scattered thunderstorm
        # 4: witness log rows
        # 5: scar pattern (random)
    ]

    SYMBOLS_BY_INTENSITY = [' ', '.', '-', '+', '*', '@', '#']

    def decode(self, z):
        pattern_idx = int(|z[0]| * 100) % 6
        intensity = z[1]
        shift = int(|z.sum()| * 100) % 4
        # ... apply pattern with intensity & shift ...
```

#### AudioHead

```python
class AudioHead:
    SR = 16000       # 16 kHz
    DURATION = 4.0   # 4 seconds

    def decode(self, z):
        f0 = 200 + (|z[0]| % 1) * 600           # fundamental 200-800 Hz
        n_harm = int(|z[1]| * 8) + 2            # 2-9 harmonics
        detune = |z[2]| * 20                    # chorus detune
        env_shape = int(|z[3]| * 4)             # 4 envelope types
        pulse_rate = 2 + int(|z[4]| * 6)        # 2-7 pulses/sec

        # synthesize
        t = linspace(0, 4.0, 64000)
        signal = sum(sin(2π * (f0*k + detune*sin(2π*0.3k*t)) * t) / k for k in range(1, n_harm+1))
        env = envelope(env_shape, pulse_rate, t)
        signal *= env
        return normalize(signal) * 0.7
```

### Step 4: Output

Each sample produces:
- `text`: doctrine-anchored prose (typically 80-200 chars)
- `image`: 16x16 ASCII art (256 chars + header)
- `audio`: 64000 float32 samples (16 kHz, 4 seconds)

The pipeline returns all three for each of n_samples latents.

## Performance

| Stage | Time | Memory |
|---|---|---|
| JEV probe | 2-5s | ~1MB |
| VAE training | <1s | <1MB |
| Sample + decode | <0.1s | ~1MB |
| **Total per lore** | **5-30s** | **~5MB** |

No GPU. No torch. No 700MB model downloads.

## Why This Design

- **Universal head**: any modality can be added by writing a `Head.decode(z)` method.
  The shared backbone (VAE) means we don't retrain when adding modalities.
- **Semantic latent**: JEV features are calibrated probabilistic values. They're
  not pixel features (BigVGAN) or SSL embeddings (Semantic-VAE) — they're
  *interpretable* semantic features.
- **Tiny VAE**: 6 dims input, 6 dims output. Trivially small. Trained on the
  fly on the input lore's features. No pre-training needed.
- **Pre-baked heads**: Templates and word pools capture doctrine vocabulary
  (witness log, canon gate, oracle, scars). No learned LM needed.

## Future Architecture

When the heads learn (instead of being pre-baked), the architecture becomes:
```
lore → JEV → 6 features → VAE.encode → z → [LM head | CNN head | WaveNet head]
                                                                     ↓
                                                              (prose | image | audio)
```

For now, the polyvocoder is a *template-based* universal head. The learned-head
version is polyformalism port 8 (planned for Oct 2026).

