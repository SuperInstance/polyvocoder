# Heads: How to Add a New Modality

The polyvocoder's "universal head" pattern means you can add ANY output modality
by writing a `Head.decode(z)` method. This document walks through adding new heads.

## Existing Heads

| Head | Output | Determinism | Complexity |
|---|---|---|---|
| TextHead | Prose (80-200 chars) | Per-z deterministic | Low (templates + pools) |
| ImageHead | 16x16 ASCII art | Per-z deterministic | Medium (6 patterns + intensity) |
| AudioHead | 4-sec 16kHz waveform | Per-z deterministic | Medium (harmonic synthesis) |

## How to Add a New Head

### Pattern: A Head Class

```python
# polyvocoder/heads/my_new_head.py

import numpy as np
from polyvocoder.heads_v2 import Head  # base class (just defines interface)


class MyVideoHead(Head):
    """Generate a 5-second 16x16 ASCII animation."""

    N_FRAMES = 60  # 12 fps for 5 seconds
    PATTERNS = [...]  # your 6 base patterns

    def decode(self, z):
        # Use z coordinates to drive animation parameters
        speed = abs(z[5]) * 2 + 0.5  # animation speed 0.5-2.5 fps
        intensity = z[1] * 2
        pattern_idx = int(abs(z[0]) * 100) % len(self.PATTERNS)

        frames = []
        for frame_idx in range(self.N_FRAMES):
            t = frame_idx / self.N_FRAMES
            shift = int(t * speed * 16) % 16
            frame = self._render_frame(self.PATTERNS[pattern_idx], shift, intensity)
            frames.append(frame)

        return frames

    def _render_frame(self, pattern, shift, intensity):
        # ... your rendering logic ...
        return ascii_frame
```

### Register in Pipeline

Add your head to `polyvocoder/pipeline_v2.py`:

```python
from polyvocoder.heads.my_new_head import MyVideoHead

# In run_pipeline_v2():
video_head = MyVideoHead()
for z in samples:
    text = text_head.decode(z)
    image = image_head.decode(z)
    audio = audio_head.decode(z)
    video = video_head.decode(z)  # NEW
    decoded.append({
        "latent": z.tolist(),
        "text": text,
        "image": image,
        "audio": audio,
        "video": video,  # NEW
    })
```

### Design Principles for New Heads

1. **Deterministic per-z**: same z → same output. (Use z for RNG seed.)
2. **Doctrinal vocabulary**: include the canonical terms (witness log, canon gate,
   oracle, substrate, FNV-1a, scars) where applicable.
3. **Bounded complexity**: avoid O(n³) or larger. The head should run in <0.1s for
   typical z.
4. **Use z coordinates as parameters**: don't add new learned parameters to the
   pipeline. The whole point of the universal head is that z is the only "knob".

## Examples of Heads You Could Build

### ColorPaletteHead

```python
class ColorPaletteHead:
    """Convert latent to 5-color palette (hex codes)."""

    def decode(self, z):
        # Each color is a 3-dim RGB
        # Sample 5 colors from different "regions" of z
        colors = []
        for i in range(5):
            start = i * 6 // 5
            r = int(255 * abs(z[start % 6]))
            g = int(255 * abs(z[(start + 1) % 6]))
            b = int(255 * abs(z[(start + 2) % 6]))
            colors.append(f"#{r:02x}{g:02x}{b:02x}")
        return colors
```

### MusicNoteHead

```python
class MusicNoteHead:
    """Convert latent to a 4-measure melody (MIDI notes)."""

    NOTES = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    DURATIONS = [0.25, 0.5, 1.0]

    def decode(self, z):
        melody = []
        for i in range(16):
            note_idx = int(abs(z[i % 6]) * 7) % 7
            octave = 4 + (int(abs(z[(i + 1) % 6]) * 2) % 2)
            duration = self.DURATIONS[int(abs(z[(i + 2) % 6]) * 3) % 3]
            melody.append({
                "note": f"{NOTES[note_idx]}{octave}",
                "duration": duration,
            })
        return melody
```

### SVGHead

```python
class SVGHead:
    """Generate a 200x200 SVG based on latent."""

    def decode(self, z):
        # Use z to position shapes
        cx = int(100 + z[0] * 80)
        cy = int(100 + z[1] * 80)
        r = int(20 + abs(z[2]) * 50)
        color = f"hsl({int(z[3] * 360)}, 70%, 50%)"

        return f"""<svg width="200" height="200">
            <circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}" />
        </svg>"""
```

### VectorHead

```python
class VectorHead:
    """Generate a 384-dim embedding vector (compatible with sentence-transformers)."""

    def decode(self, z):
        # Combine z with deterministic noise to get 384 dims
        rng = np.random.default_rng(int(abs(z.sum()) * 1000) % 2**32)
        # Use z to modulate
        base = rng.uniform(-1, 1, 384)
        # Mix in z for canon-aware flavor
        for i in range(min(6, len(z))):
            base[i::6] *= z[i]
        # Normalize
        return base / np.linalg.norm(base)
```

## Testing Your Head

Write a smoke test in `tests/test_my_head.py`:

```python
import sys
import numpy as np
sys.path.insert(0, "/path/to/polyvocoder")

from polyvocoder.heads.my_new_head import MyVideoHead


def test_my_video_head():
    rng = np.random.default_rng(42)
    z = rng.uniform(-1, 1, 6)
    output = MyVideoHead().decode(z)
    assert isinstance(output, list)
    assert len(output) == 60  # N_FRAMES
    assert all(isinstance(f, str) for f in output)


def test_my_video_head_variation():
    rng = np.random.default_rng(42)
    z1 = rng.uniform(-1, 1, 6)
    z2 = rng.uniform(-1, 1, 6)
    o1 = MyVideoHead().decode(z1)
    o2 = MyVideoHead().decode(z2)
    assert o1 != o2


if __name__ == "__main__":
    test_my_video_head()
    test_my_video_head_variation()
    print("MyVideoHead tests passed!")
```

## Submitting Your Head

Once you've built a head you'd like to share:

1. Fork `SuperInstance/polyvocoder`
2. Add your head to `polyvocoder/heads/` (or `polyvocoder/heads_v2.py`)
3. Write tests in `tests/test_<your_head>.py`
4. Update `docs/HEADS.md` with your head's design
5. Open a PR with a sample output (image, audio, etc.)

We'll merge if your head is doctrine-aligned (uses witness log, canon gate,
oracle vocabulary where applicable) and deterministic per-z.

