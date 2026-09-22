# Polyvocoder

A 200-line **universal head** for the Quilt substrate walker.

Encode any substrate cell (lore, polygon, audio rhythm) via JEV → semantic features →
VAE-compress to a shared latent space → decode to any output modality (prose, image,
audio).

## Architecture

```
lore → JEV scores (canonical questions) → VAE.encode → z (latent)
                                                  ↓
                              VAE.decode → output head → modality
                              (text / image / audio)
```

The polyvocoder is the *fusion* of BigVGAN's universality (any in → any out via shared
backbone + typed head) and Semantic-VAE's semantic latent space (SSL extractor → VAE).

## Files

- `jev_extractor.py` — JEV client that scores a text input across canonical questions
- `vae.py` — numpy-only VAE that compresses JEV scores to latent and decodes back
- `heads.py` — Pluggable output heads: text head, image head, audio head
- `pipeline.py` — End-to-end: encode lore → VAE → sample → decode to all 3 modalities
- `cli.py` — CLI: `python -m polyvocoder "seed for canon"` runs the pipeline
- `tests/test_smoke.py` — pytest that runs end-to-end with one lore
- `pyproject.toml` — minimal packaging
- `LICENSE` — MIT

## Quick Start

```bash
pip install requests numpy  # only deps
cd polyvocoder
export TYPESAFEAI_KEY=<your-key>
python -m polyvocoder "The substrate walker is a 4D cell graph..."
```

## Why?

Casey asked for it on Sept 22, 2026 ("could we build a polyvocoder that fuses the
BigVGAN + Semantic-VAE ideas?"). This is the answer.
