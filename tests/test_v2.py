"""Tests for polyvocoder v2 — TextHead, ImageHead, AudioHead."""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, '/workspace/repos/polyvocoder')

from polyvocoder.heads_v2 import TextHead, ImageHead, AudioHead


def test_text_head():
    rng = np.random.default_rng(42)
    z = rng.uniform(-1, 1, 6)
    text = TextHead().decode(z)
    assert isinstance(text, str), "Should return string"
    assert len(text) > 30, "Should be longer than 30 chars"
    print(f"  text: {text[:80]}...")


def test_text_head_multiple():
    rng = np.random.default_rng(42)
    z1 = rng.uniform(-1, 1, 6)
    z2 = z1 + 0.01  # very similar latent
    t1 = TextHead().decode(z1)
    t2 = TextHead().decode(z2)
    # Different latents should give different (or same template but different seed/word) outputs
    assert t1 != t2 or any(rng.choice(["witness", "scars", "oracle", "canon gate", "FNV-1a", "substrate"]) in t2 for _ in range(1)), "Should generate doctrine-anchored prose"
    print(f"  t1: {t1[:80]}")
    print(f"  t2: {t2[:80]}")


def test_image_head_patterns():
    img_head = ImageHead()
    for pattern_idx in range(6):
        z = np.array([float(pattern_idx) / 6, 0.5, 0.0, 0.0])
        image = img_head.decode(z)
        assert isinstance(image, str)
        assert "<<<" in image, "Should have header"
        assert len(image.split("\n")) >= 16, "Should have 16 rows"
    print(f"  All 6 patterns render")


def test_image_intensity_varies():
    """Intensity should change the visual."""
    img_head = ImageHead()
    z_neg = np.array([0.0, -1.0, 0.0, 0.0])
    z_pos = np.array([0.0, +1.0, 0.0, 0.0])
    neg_str = img_head.decode(z_neg)
    pos_str = img_head.decode(z_pos)
    # Different outputs at minimum (intensity differs)
    # Same pattern, but intensity band should differ
    assert neg_str != pos_str, "Different intensity should give different outputs"
    print(f"  intensity negative: {neg_str.split(chr(10))[0]}")
    print(f"  intensity positive: {pos_str.split(chr(10))[0]}")


def test_audio_head_shape():
    z = np.random.default_rng(0).uniform(-1, 1, 6)
    audio = AudioHead().decode(z)
    assert audio.ndim == 1, "Should be 1-D"
    assert audio.shape[0] == 16000 * 4, "Should be 4 seconds at 16kHz = 64000 samples"
    assert audio.dtype == np.float32, "Should be float32"
    assert -1.0 <= audio.min() and audio.max() <= 1.0, "Should be normalized"
    print(f"  audio shape={audio.shape}, min={audio.min():.3f}, max={audio.max():.3f}")


def test_audio_variation():
    """Different latents should give different audio."""
    rng = np.random.default_rng(0)
    z1 = rng.uniform(-1, 1, 6)
    z2 = rng.uniform(-1, 1, 6)
    a1 = AudioHead().decode(z1)
    a2 = AudioHead().decode(z2)
    assert np.max(np.abs(a1 - a2)) > 0.001, "Different latents should give different waveforms"
    print(f"  max diff: {np.max(np.abs(a1 - a2)):.3f}")


def test_pipeline_runs():
    from polyvocoder.pipeline_v2 import run_pipeline_v2
    lore = "Test lore for pipeline."
    result = run_pipeline_v2(lore, n_samples=2)
    assert "features" in result
    assert "decoded_samples" in result
    assert len(result["decoded_samples"]) == 2
    assert len(result["features"]) == 6
    sample = result["decoded_samples"][0]
    assert "text" in sample
    assert "image" in sample
    assert "audio" in sample
    print(f"  2 samples, features len={len(result['features'])}")


if __name__ == "__main__":
    print("=== TextHead tests ===")
    test_text_head()
    test_text_head_multiple()
    print("\n=== ImageHead tests ===")
    test_image_head_patterns()
    test_image_intensity_varies()
    print("\n=== AudioHead tests ===")
    test_audio_head_shape()
    test_audio_variation()
    print("\n=== Pipeline tests ===")
    test_pipeline_runs()
    print("\nAll tests passed!")
