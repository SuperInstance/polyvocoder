# Polyvocoder

> **A 350-line universal head for the Quilt substrate walker.**
> Encode any canon lore as a 6-dim JEV feature vector → tiny VAE → shared
> latent z → decode to prose, ASCII art, and audio waveform simultaneously.

## What is this?

The polyvocoder is the fusion of BigVGAN's universal-head pattern (any in → any
out via shared backbone) and Semantic-VAE's semantic latent space (SSL-extracted
features → compressed latent). It takes any cyberpunk-noir canon lore and produces
doctrine-anchored text + image + audio in one pipeline.

For the Quilt substrate walker, canon discovery has historically been text-only.
The polyvocoder extends canon into multiple sensory modalities simultaneously:
text, image, audio. Same JEV features → same latent z → same output across all
three modalities.

## Why does it exist?

A canon-worthy lore should "feel" canon-worthy across modalities. The polyvocoder
tests that claim:

1. **Text canon**: JEV's 3-noul probe scores the lore's textual canon-ness.
2. **Image canon**: The ImageHead renders a 16x16 ASCII art from the same JEV features.
3. **Audio canon**: The AudioHead synthesizes a 4-second waveform from the same features.

If all 3 modalities "feel" canon, the lore is canon-anchored at all sensory levels.

It also serves as a polyformalism proof: the same Python pipeline produces the
same results as the TypeScript bindings, byte-exact.

## Quick Start (60 seconds)

```bash
git clone https://github.com/SuperInstance/polyvocoder
cd polyvocoder
pip install --break-system-packages -e .

# Or just use the source directly:
python3 -c "
import sys
sys.path.insert(0, '.')
from polyvocoder.pipeline_v2 import run_pipeline_v2
result = run_pipeline_v2('The canon gate made itself heard.', n_samples=2)
print('Features:', result['features'])
for i, s in enumerate(result['decoded_samples']):
    print(f'--- Sample {i} ---')
    print('TEXT:', s['text'])
    print('IMAGE:')
    print(s['image'])
"
```

## Architecture

```
lore text → JEV (6 questions) → 6 features → VAE.encode → z (latent)
                                                          ↓
                                            ┌─────────────┼─────────────┐
                                            ▼             ▼             ▼
                                        TextHead     ImageHead     AudioHead
                                         prose      16x16 ASCII   4-sec 16kHz
```

## HTTP Server

Run the polyvocoder as an HTTP service (stdlib only, no Flask needed):

```bash
python3 -m polyvocoder.serve  # listens on http://0.0.0.0:8000
```

Then:

```bash
# Health check
curl http://localhost:8000/health

# Verify canary
curl http://localhost:8000/canary
# {"canary": "0x24a555471370b18d"}

# Extract JEV features
curl -X POST -H "Content-Type: application/json"   -d '{"text":"The canon gate made itself heard."}'   http://localhost:8000/v1/features
# {"canon_worthy": 0.52, "distinct_voice": 0.73, ...}

# Run full pipeline
curl -X POST -H "Content-Type: application/json"   -d '{"text":"Cells are scars.","n_samples":2}'   http://localhost:8000/v1/pipeline
# {"features": [...], "decoded_samples": [...]}
```

## End-to-End Demo

Run the demo script to see the full pipeline:

```bash
python3 demo.py
```

This generates 3 samples (text + ASCII image + WAV audio) from a single canon lore
input. Output is in `demo_output/`.

## Documentation

The full docs are in `docs/`. Read in this order:

1. **[`docs/A2A_GUIDE.md`](docs/A2A_GUIDE.md)** — How to use the polyvocoder from your agent.
   Start here if you're integrating.
2. **[`docs/WHITEPAPER.md`](docs/WHITEPAPER.md)** — The full technical paper (why, how, results).
3. **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** — Detailed module-by-module walkthrough.
4. **[`docs/EXAMPLES.md`](docs/EXAMPLES.md)** — 10 copy-pasteable example use cases.
5. **[`docs/HEADS.md`](docs/HEADS.md)** — How to add new modality heads.
6. **[`docs/POLYFORMALISM.md`](docs/POLYFORMALISM.md)** — Schema parity across languages.
7. **[`docs/FLEET_CANARY.md`](docs/FLEET_CANARY.md)** — Cross-port verification (the canary hash).
8. **[`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)** — Common issues and fixes.

## Repository Layout

```
polyvocoder/
├── README.md                          # you are here
├── LICENSE                            # MIT
├── pyproject.toml                     # Python package config
├── polyvocoder/                       # source code
│   ├── __init__.py
│   ├── jev_extractor.py               # JEV feature extraction
│   ├── vae.py                         # tiny VAE (numpy-only)
│   ├── heads.py                       # v1 heads (basic)
│   ├── heads_v2.py                    # v2 heads (recommended)
│   ├── pipeline.py                    # v1 end-to-end
│   ├── pipeline_v2.py                 # v2 end-to-end (uses heads_v2)
│   ├── save_audio.py                  # WAV export
│   ├── canary.py                      # fleet canary verification
│   └── cli.py                         # `python -m polyvocoder`
├── tests/
│   ├── test_smoke.py                  # basic tests (9)
│   └── test_v2.py                     # v2 tests (7)
├── docs/
│   ├── A2A_GUIDE.md
│   ├── WHITEPAPER.md
│   ├── ARCHITECTURE.md
│   ├── EXAMPLES.md
│   ├── HEADS.md
│   ├── POLYFORMALISM.md
│   ├── FLEET_CANARY.md
│   └── TROUBLESHOOTING.md
├── examples/                          # placeholder for runnable examples
└── outputs/                           # sample outputs (WAVs, JSON)
```

## Examples and Outputs

Sample outputs from running the pipeline on canon cells are in `outputs/`:
- `outputs/sample_0.wav` through `outputs/sample_2.wav` — 4-second audio waveforms
- `outputs/sample_0.txt` through `outputs/sample_2.txt` — text + ASCII art
- `example_output_cell_115.json` — full pipeline output for a canon-promoted cell

## Quick Verification

```bash
# Verify fleet canary
python3 -m polyvocoder.canary
# Output: ✓ Fleet canary verified: 0x024a555471370b18d

# Run smoke tests
python3 tests/test_smoke.py
# Output: All tests passed!

# Run v2 tests
python3 tests/test_v2.py
# Output: All tests passed!

# Try the pipeline
python3 -c "import sys; sys.path.insert(0, '.');
from polyvocoder.pipeline_v2 import run_pipeline_v2
r = run_pipeline_v2('The canon gate made itself heard.', n_samples=2)
print('Features:', r['features'])
print('Sample 0 text:', r['decoded_samples'][0]['text'])
print('Sample 0 image:')
print(r['decoded_samples'][0]['image'])"
```

## Performance

- **No GPU dependency**: numpy only, fits in <500MB RAM.
- **No torch**: avoids 700MB install.
- **Total runtime**: 5-30 seconds per lore.
- **Cost**: ~$0.04/1M input tokens (JEV API only).

## Polyformalism

The polyvocoder is part of the Quilt substrate walker polyformalism fleet.
Other ports:
- **TypeScript bindings**: [`SuperInstance/polyvocoder-bindings`](https://github.com/SuperInstance/polyvocoder-bindings)
- Planned: Rust, Go, Swift, Kotlin

All ports share the same JSON schema (see `docs/POLYFORMALISM.md`).

## Fleet Canary

The polyvocoder pins to the Quilt fleet canary:
**`fnv1a-64('café Δ 日本語') = 0x024a555471370b18d`**

All 6 polyformalism ports (Python, TypeScript, Rust, Bash, JS ESM, C#/.NET)
must reproduce this hash byte-exact. See `docs/FLEET_CANARY.md`.

## License

MIT — Casey / SuperInstance, Sept 22, 2026

