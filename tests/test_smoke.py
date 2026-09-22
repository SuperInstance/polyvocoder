"""Smoke test: run the polyvocoder end-to-end on a single lore.

Skips if TYPESAFEAI_KEY is not set (JEV API unavailable).
"""
import os
import sys
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from polyvocoder.jev_extractor import extract_jev_features, DEFAULT_QUESTIONS
from polyvocoder.vae import VAE
from polyvocoder.heads import HEADS, decode_all_modalities
from polyvocoder.pipeline import run_pipeline


def test_jev_extractor_with_key():
    """JEV extractor returns a fixed-dim feature vector when key is set."""
    if not os.environ.get("TYPESAFEAI_KEY"):
        pytest.skip("TYPESAFEAI_KEY not set")
    text = "The substrate walker is a 4D cell graph."
    feats = extract_jev_features(text)
    assert len(feats) == len(DEFAULT_QUESTIONS)
    for f in feats:
        assert 0.0 <= f <= 1.0


def test_jev_extractor_no_key():
    """JEV extractor raises when key is missing."""
    if os.environ.get("TYPESAFEAI_KEY"):
        pytest.skip("TYPESAFEAI_KEY is set")
    with pytest.raises(ValueError):
        extract_jev_features("test")


def test_vae_forward_shape():
    """VAE encode/decode preserves input shape."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, (1, 8)).astype(np.float32)
    vae = VAE(input_dim=8, latent_dim=4, hidden_dim=16)
    recon, mu, log_sigma = vae.forward(x)
    assert recon.shape == x.shape
    assert mu.shape == (1, 4)
    assert log_sigma.shape == (1, 4)


def test_vae_kl_decreases():
    """VAE fit should decrease total loss over training."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (50, 8)).astype(np.float32)
    vae = VAE(input_dim=8, latent_dim=4, hidden_dim=16)
    losses = vae.fit(X, n_epochs=30)
    # First loss should be larger than last
    assert losses[-1] < losses[0]


def test_text_head_runs():
    """Text head produces a non-empty string from any latent."""
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, 16).astype(np.float32)
    text = HEADS["text"].decode(z)
    assert isinstance(text, str)
    assert len(text) > 50


def test_image_head_runs():
    """Image head produces a 16x16 ASCII grid."""
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, 16).astype(np.float32)
    img = HEADS["image"].decode(z)
    assert isinstance(img, str)
    lines = img.split("\n")
    assert len(lines) == 16
    for line in lines:
        assert len(line) == 16


def test_audio_head_runs():
    """Audio head produces a numpy waveform."""
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, 16).astype(np.float32)
    audio = HEADS["audio"].decode(z, n_samples=1600)
    assert audio.shape == (1600,)
    assert -1.0 <= audio.min() <= 1.0
    assert -1.0 <= audio.max() <= 1.0


def test_pipeline_end_to_end():
    """Pipeline runs end-to-end (uses JEV if key set, otherwise falls back to zeros)."""
    text = "The substrate walker is a 4D cell graph where cells are scars."
    result = run_pipeline(text, n_samples=2, verbose=False)
    assert "features" in result
    assert "mu" in result
    assert "decoded_samples" in result
    assert len(result["decoded_samples"]) == 2
    for sample in result["decoded_samples"]:
        assert "text" in sample
        assert "image" in sample
        assert "audio" in sample


def test_decode_all_modalities():
    """Decode to all modalities produces a dict with text/image/audio."""
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, 16).astype(np.float32)
    out = decode_all_modalities(z)
    assert "text" in out
    assert "image" in out
    assert "audio" in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
