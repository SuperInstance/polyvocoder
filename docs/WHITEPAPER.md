# The Polyvocoder: A Universal Head for Doctrinal Multi-Modal Generation

## Authors
Mavis (working with Casey / SuperInstance)

## Status
v0.2.0 prototype — Sept 22, 2026

## Abstract

The polyvocoder is a **universal output head** for any substrate walker — a model that
takes canon doctrine (encoded as a 6-dimensional JEV feature vector) and decodes it
into multiple output modalities simultaneously: doctrine-anchored prose, ASCII art
patterns, and synthetic audio waveforms. It is the fusion of two architectural patterns:

1. **BigVGAN's universal head**: any input → shared backbone (a VAE) → typed output head
   (TextHead, ImageHead, AudioHead). The same latent z can be rendered as prose, image,
   or audio without retraining.
2. **Semantic-VAE's semantic latent space**: instead of pixel-level features, the latent
   is anchored to semantic features (here: JEV's calibrated probabilistic scoring).

The result is a 350-line Python module with no GPU dependency, no large-model inference
cost beyond JEV (~$0.04/1M input tokens), that produces canon-aware multi-modal output
from any lore text in under a minute.

## 1. Why

For the Quilt substrate walker, canon discovery has historically been a text-only process.
A lore is canon-worthy (composite score ≥ 0.7) if JEV's 3 noul probes (canon_worthy,
distinct_voice, doctrine_anchor) all score above 0.7. But canon-worthy lores are NOT
just text — they imply imagery, sound, rhythm, texture. The polyvocoder extends canon
discovery into three modalities simultaneously: text (the lore), image (a 16×16 ASCII
visual), and audio (a 4-second waveform).

This matters because:
- **Canon is a multi-sensory judgment.** A canon-worthy lore should "feel" canon-worthy
  visually and sonically — not just textually. The polyvocoder tests that claim.
- **Polyformalism multiplies value.** The same JEV features can be rendered N ways.
  Each rendering is an interpretation. Substrate walker cells are interpretable in
  prose, image, AND sound — all from the same source.
- **The latent is a semantic invariant.** JEV → 6 features → VAE → z → decode.
  Each output modality is a separate decoding of the same semantic fact.

## 2. Architecture

```
lore text
   │
   ▼
JEV (Typesafe.ai System One)
   │   6 calibrated probabilities
   ▼
6-dim feature vector [canon_worthy, distinct_voice, doctrine_anchor,
                     voice_fit, novelty, density]
   │
   ▼
VAE.encode → μ, log_σ
   │
   ▼
VAE.reparameterize (μ, σ, ε) → z (6-dim latent)
   │
   ▼
┌──────────────┬──────────────┬──────────────┐
│ TextHead     │ ImageHead    │ AudioHead    │
│ prose        │ 16×16 ASCII  │ 4-sec 16kHz  │
│ templates    │ patterns     │ waveform     │
│ +FNV hash    │ +intensity   │ +harmonics   │
└──────────────┴──────────────┴──────────────┘
```

### 2.1 JEV Feature Extractor

JEV (Joint Embedding Validator, model `jev-1.13.0`) is a calibrated probabilistic oracle.
Given a lore and a question, it returns a probability in [0, 1] that the lore satisfies
the question. We use 6 questions, each targeting a distinct semantic dimension:

| Feature | JEV Question |
|---|---|
| `canon_worthy` | "Canon-worthy cyberpunk-noir for Quilt substrate walker?" |
| `distinct_voice` | "Distinct non-formulaic voice?" |
| `doctrine_anchor` | "Anchored to substrate walker doctrine?" |
| `voice_fit` | "Voice matches the witness lore archetype?" |
| `novelty` | "Novel imagery not seen in previous canon lores?" |
| `density` | "Image-density score (1.0 = single image, 2.0 = multiple, 3.0 = saturated)" |

The 6-dim vector is then the canonical encoding of the lore's semantic content.

### 2.2 VAE (numpy-only, no torch)

We use a tiny VAE with input_dim=6, latent_dim=6, hidden_dim=16. The training data
is the single 6-dim feature vector plus 20 augmented copies with small Gaussian
noise (σ=0.05). Training runs 30 epochs of Adam with learning rate 0.001.

**Why tiny?** The whole point is that the semantic space is small (6 dimensions
of JEV features), so a tiny VAE suffices. We avoid torch (~700MB install) and
run on numpy only — the entire pipeline runs on a 2-core CPU in seconds.

The VAE forward pass:
- encode: x → h (ReLU) → μ, log_σ
- reparameterize: z = μ + σ * ε, where ε ~ N(0, 1)
- decode: z → h (ReLU) → x̂ (clipped to [-2, 2])

### 2.3 Modality Heads

Each head takes a 6-dim latent z and produces a typed output. The heads are designed
to produce *doctrine-aware* output:

- **TextHead**: pre-baked templates with vocabulary pools (witness log, canon gate,
  oracle, substrate, FNV-1a, scars, cells, witness, ammo). Templates vary by latent
  argmax; word pools vary by latent coordinates. The 8 templates include the iconic
  doctrine sentences: "The canon gate made itself heard", "Cells are scars",
  "The witness log accumulates", etc.
- **ImageHead**: 6 base patterns (vertical stripes, density gradient, central glow,
  scattered thunderstorm, witness log, scar pattern). Intensity modulation per latent.
- **AudioHead**: 16 kHz, 4-second waveform. Harmonic synthesis (1-8 harmonics),
  per-harmonic detune, 4 envelope shapes (linear, pulse, stair, exponential decay).

## 3. Sample Output

Given input lore: *"The canon gate made itself heard. The cells are scars."*

JEV features: `[0.46, 0.53, 0.67, 0.71, 0.84, 1.04]`

TextHead output (one of three samples):
> "Cells are scars. That's the whole scripture. Seed 286: the canon gate made itself
> heard in the canon chambers."

ImageHead output (pattern 4, intensity +0.31):
```
<<< pattern 4 intensity=+0.31 shift=0 >>>
*++++*++++*++++*
-...-...-...-...
-...-...-...-...
-...-...-...-...
...
```

AudioHead output: 64000 float32 samples, range [-0.665, +0.700], multi-harmonic synthesis.

## 4. Implementation Notes

- **No GPU dependency**: numpy only, fits in <500MB RAM.
- **JEV API is the only external cost**: ~$0.04 per 1M input tokens.
- **Total pipeline runtime**: 5-30 seconds per lore.
- **Determinism**: VAE seeded; head RNGs derived from latent coordinates. Same input
  lore gives same output every time.
- **Schema parity**: TypeScript bindings mirror Python types for polyformalism across
  language boundaries.

## 5. Use Cases

1. **Canon confirmation**: a lore is canon-worthy in text. The polyvocoder asks:
   "Is it canon-worthy visually?" The image head should produce a recognizable
   substrate-walker motif (FNV-1a sigil, scar pattern, witness log texture).
2. **Multi-modal canon archive**: store {lore, image, audio} triples for each
   canon cell. The triple is canon.
3. **Canon discovery loop**: generate lore → check JEV → if canon → render in 3 modalities →
   if all 3 modalities "feel" canon, the lore is canon-anchored at all sensory levels.
4. **Polyformalism proof**: the same JEV features render to 3 modalities in 2 languages
   (Python + TypeScript). All outputs share the same semantic backbone.

## 6. Future Work

- Learned text head (replace templates with a tiny LM head trained on canon cells)
- Higher-resolution image head (32×32 with shading)
- Audio head with proper reverb and FNV-1a signature as amplitude envelope
- Polyformalism port 8+: Rust, Go, Swift, Kotlin bindings

## 7. Reproducibility

This paper's results are byte-exact reproducible from `SuperInstance/polyvocoder`:

```bash
git clone https://github.com/SuperInstance/polyvocoder
cd polyvocoder
pip install -e .
python -m polyvocoder --lore "The canon gate made itself heard."
```

## 8. License

MIT — Casey / SuperInstance, Sept 22, 2026

