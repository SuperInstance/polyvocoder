# Troubleshooting Guide

Common issues and how to fix them.

## Installation

### Issue: `pip install` fails with 503 errors

**Cause**: PyPI is intermittent. Sometimes the package mirror is offline.

**Fix**:
```bash
pip install --break-system-packages --index-url https://pypi.org/simple/ numpy requests
```

If that fails too, try installing numpy first (it's the only hard dependency):
```bash
pip install --break-system-packages numpy
pip install --break-system-packages requests
```

### Issue: `ModuleNotFoundError: No module named 'polyvocoder'`

**Cause**: The polyvocoder package isn't installed. You might have downloaded the
code but not run `pip install -e .`

**Fix**:
```bash
cd /path/to/polyvocoder
pip install --break-system-packages -e .
```

Or, if you want to use the code without installing:
```python
import sys
sys.path.insert(0, "/path/to/polyvocoder")
from polyvocoder.pipeline_v2 import run_pipeline_v2
```

## JEV API Issues

### Issue: `requests.exceptions.ConnectionError` or 503

**Cause**: Typesafe.ai API has DNS-cache-overflow issues (similar to other
LLM providers). ~10-30% of calls fail transiently.

**Fix**: Add retry-with-backoff. The polyvocoder doesn't include this by default,
but here's a wrapper:

```python
import time
import requests

def jev_with_retry(state, questions, max_retries=5, base_delay=2):
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                "https://api.typesafe.ai/v1/systemone",
                headers={"Authorization": f"Bearer {os.environ['TYPESAFEAI_KEY']}"},
                json={"state": state, "questions": questions},
                timeout=120,
            )
            if resp.status_code == 200:
                return resp.json()
        except requests.exceptions.RequestException:
            pass
        if attempt < max_retries - 1:
            time.sleep(base_delay * (attempt + 1))
    raise RuntimeError(f"JEV failed after {max_retries} retries")
```

### Issue: `KeyError: 'answers'`

**Cause**: JEV API returns an error response with no `answers` key.

**Fix**: Check the response status and content:
```python
resp = requests.post(...)
print(f"Status: {resp.status_code}")
print(f"Body: {resp.text[:200]}")
if resp.status_code != 200:
    print(f"JEV error: {resp.text}")
```

Common errors:
- 401: bad API key
- 422: malformed questions (e.g., missing `criteria` for choice questions)
- 429: rate limited (slow down)
- 503: server overload (retry)

### Issue: All features are 0.5 (no discrimination)

**Cause**: The lore is too short or too generic for JEV to extract distinct features.

**Fix**: Use a longer, more specific lore. The polyvocoder needs at least 50
characters of substantive content.

```python
# Too short — JEV can't distinguish
lore = "The gate opened."

# Better — gives JEV something to work with
lore = "The canon gate made itself heard at midnight. Cells are scars. That's the whole scripture."
```

### Issue: JEV API key not found

**Cause**: The `TYPESAFEAI_KEY` environment variable is not set.

**Fix**:
```bash
export TYPESAFEAI_KEY="your-key-here"
```

Get a key at https://typesafe.ai. The API is pay-per-use.

## VAE Issues

### Issue: VAE loss is NaN after a few epochs

**Cause**: Numerical gradient descent with too-large epsilon or unstable inputs.

**Fix**: The default epsilon is 1e-3. If you're getting NaN, try:
1. Smaller epsilon: `eps=1e-4` (in `_compute_grads`)
2. Lower learning rate: `lr=0.0001`
3. Gradient clipping: clip gradients to [-1, 1] before update

```python
# In vae.py, modify _compute_grads:
def _compute_grads(self, x, lr=0.001, kl_weight=0.01):
    eps = 1e-4  # was 1e-3
    # ...
```

### Issue: VAE produces identical outputs for different inputs

**Cause**: KL weight is too high, latent is collapsing to N(0, 1).

**Fix**: Reduce KL weight:
```python
vae.fit(X, kl_weight=0.001)  # was 0.01
```

### Issue: VAE training is slow (>10 seconds)

**Cause**: Numerical gradients are expensive (~10 dimensions per parameter).

**Fix**: 
1. Reduce hidden_dim: `hidden_dim=8` (was 16)
2. Fewer epochs: `n_epochs=10` (was 30)
3. Use analytical gradients (TODO: not implemented yet)

For larger VAEs, switch to PyTorch:
```python
import torch
import torch.nn as nn

class TorchVAE(nn.Module):
    # ... use torch.nn.Linear, torch.optim.Adam ...
```

## Head Issues

### Issue: TextHead produces same output for all samples

**Cause**: The latent z is dominated by VAE mean (μ), not by reparameterized noise.

**Fix**: Sample with more noise:
```python
z = mu + sigma * epsilon
# Where epsilon is large:
epsilon = np.random.normal(0, 2.0, size=mu.shape)  # bigger noise
z = mu + sigma * epsilon
```

### Issue: ImageHead patterns all look the same

**Cause**: Latent doesn't vary across samples.

**Fix**: Same as TextHead. Force more noise in reparameterization.

### Issue: AudioHead output sounds like a single sine wave

**Cause**: Only one harmonic is being generated. Check `n_harm`:
```python
print(f"n_harm: {int(abs(z[1]) * 8) + 2}")
# If this is consistently 2, increase the multiplier:
# Change `abs(z[1]) * 8` to `abs(z[1]) * 12`
```

### Issue: WAV file won't play

**Cause**: Header is wrong, or audio is clipped.

**Fix**: Use `polyvocoder/save_audio.py:save_wav()` which handles format correctly:
```python
from polyvocoder.save_audio import save_wav
save_wav(audio_array, "output.wav")  # this is correct
```

If you're writing WAV manually, use 16-bit PCM, mono, 16kHz:
```python
import wave
audio_int16 = (audio * 32767).astype(np.int16)
with wave.open(path, "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)  # 16-bit
    wf.setframerate(16000)
    wf.writeframes(audio_int16.tobytes())
```

## Cross-Project Issues

### Issue: Different results in different runs

**Cause**: VAE random initialization, head RNGs, JEV API state.

**Fix**: Set all seeds:
```python
import numpy as np
np.random.seed(42)
result = run_pipeline_v2(lore, vae_seed=42)
```

### Issue: Different results across languages (polyformalism)

**Cause**: Different FNV-1a implementations or unicode handling.

**Fix**: Verify the fleet canary:
```python
from polyvocoder.canary import canary_hash
assert canary_hash() == 0x024a555471370b18d
```

If it doesn't match, your Python version has different unicode handling for the
canary string. Update to Python 3.11+ with `errors="surrogatepass"` and
`.replace("\x00", "")` on the encoded bytes.

## Getting Help

If your issue isn't listed here:

1. Check the GitHub issues: https://github.com/SuperInstance/polyvocoder/issues
2. Read the whitepaper: `docs/WHITEPAPER.md`
3. Read the architecture: `docs/ARCHITECTURE.md`
4. Open a new issue with a minimal reproduction

For urgent issues, contact Casey via the substrate walker protocol.

