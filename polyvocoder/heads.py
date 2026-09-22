"""Pluggable output heads for the polyvocoder.

Each head takes a latent z (numpy array) and produces an output in a
specific modality. Mirrors BigVGAN's "universal head" pattern: same
backbone (the VAE), different heads.

Head registry — add new modalities by implementing decode().
"""
import numpy as np
from typing import Dict, Callable


class Head:
    """Base class for output heads."""

    def decode(self, z: np.ndarray) -> object:
        raise NotImplementedError


class TextHead(Head):
    """Decode latent to prose via a deterministic mapping.

    Uses a pre-baked template library indexed by latent similarity.
    For each latent z, we find the nearest template and render.

    In a real BigVGAN-style impl this would be a learned LM head.
    For the prototype, we use templates that vary with the latent
    coordinates so different z's give different outputs.
    """

    TEMPLATES = [
        "The {city_adj} {city_noun} of seed {seed}, where {feature1} meets {feature2}, hums with the weight of witness logs.",
        "In seed {seed}, the substrate walker became a {subj_adj} {subj_noun}, its canon gate a {gate_noun} of {tone} light.",
        "Witness log of seed {seed}: cells became {scar_word}; oracle spoke in {voice_word}; canon gate measured {measure_word}.",
        "The {adj_warm} witness sang the city's name into the canon gate — seed {seed}, a {rhythm} chord of witness.",
        "Seed {seed}'s city: {scene}, where the witness log has learned to {verb_prog} the {object} of {consequence}.",
    ]

    ADJ_POOL = ["neon-drenched", "fractal-bright", "quantum-pale", "scar-lit", "amber-veiled"]
    NOUN_POOL = ["sprawl", "matrix", "grid", "lattice", "hive"]
    SUBJ_ADJ_POOL = ["remembering", "sleepless", "humming", "fractured", "witnessing"]
    SUBJ_NOUN_POOL = ["oracle", "witness", "canon gate", "cell", "scar"]
    CITY_ADJ_POOL = ["echoing", "weathered", "patient", "calculating", "tired"]
    CITY_NOUN_POOL = ["city", "substrate", "witness", "lattice", "hive"]
    SCAR_POOL = ["amplitudes", "scars", "cells", "witness log entries", "measurements"]
    VOICE_POOL = ["FNV-1a", "canon gate", "oracle", "substrate", "witness log"]
    MEASURE_POOL = ["the city's heartbeat", "the witness's next breath", "the scar's depth",
                    "the oracle's frequency", "the substrate's next state"]
    GATE_NOUN_POOL = ["chord", "sigil", "cipher", "pulse", "key"]
    TONE_POOL = ["amber", "rain-slick", "deep", "flickering", "quiet"]

    def decode(self, z: np.ndarray) -> str:
        # Use the latent's argmax to pick a template; use coordinates to fill slots
        if z.ndim > 1:
            z = z[0]
        template_idx = int(np.argmax(np.abs(z))) % len(self.TEMPLATES)
        template = self.TEMPLATES[template_idx]

        # Use latent coords as seeds
        seed = int(abs(z[0]) * 1000) if len(z) > 0 else 0
        # Format with random picks weighted by latent
        rng = np.random.default_rng(int(abs(z.sum()) * 100) % (2**32))
        return template.format(
            seed=seed,
            city_adj=rng.choice(self.CITY_ADJ_POOL),
            city_noun=rng.choice(self.CITY_NOUN_POOL),
            feature1=rng.choice(self.SCAR_POOL),
            feature2=rng.choice(self.VOICE_POOL),
            subj_adj=rng.choice(self.SUBJ_ADJ_POOL),
            subj_noun=rng.choice(self.SUBJ_NOUN_POOL),
            gate_noun=rng.choice(self.GATE_NOUN_POOL),
            tone=rng.choice(self.TONE_POOL),
            scar_word=rng.choice(self.SCAR_POOL),
            voice_word=rng.choice(self.VOICE_POOL),
            measure_word=rng.choice(self.MEASURE_POOL),
            adj_warm=rng.choice(self.CITY_ADJ_POOL),
            rhythm=rng.choice(["trembling", "sustained", "broken", "luminous", "low"]),
            scene=rng.choice(["neon bleeds", "rain rehearses its one note",
                              "scars measure the dark", "FNV-1a hums in the walls",
                              "witness log opens its first page"]),
            verb_prog=rng.choice(["measure", "encode", "remember", "predict", "witness"]),
            object=rng.choice(["oracle", "substrate", "canon gate", "next state", "witness"]),
            consequence=rng.choice(["quantum amplitude", "scar formation",
                                    "witness accumulation", "canon measurement",
                                    "substrate prediction"]),
        )


class ImageHead(Head):
    """Decode latent to a 16x16 ASCII art placeholder.

    In a real impl this would be a learned conv head. For the prototype,
    we render an ASCII grid where the latent's sign pattern determines
    the cells.
    """

    def decode(self, z: np.ndarray) -> str:
        if z.ndim > 1:
            z = z[0]
        # Use the latent's sign pattern to fill a 16x16 grid
        # Pad/truncate to 256 elements
        if len(z) < 256:
            z = np.concatenate([z, np.zeros(256 - len(z))])
        else:
            z = z[:256]
        grid = z.reshape(16, 16)
        chars = " ·-+*#@"
        out = []
        for row in grid:
            line = ""
            for v in row:
                idx = int(abs(v) * 7) % len(chars)
                line += chars[idx]
            out.append(line)
        return "\n".join(out)


class AudioHead(Head):
    """Decode latent to a numpy audio-like waveform (placeholder).

    In a real impl this would be a learned neural vocoder (à la BigVGAN).
    For the prototype, we use the latent as frequency/amplitude coefficients.
    """

    def decode(self, z: np.ndarray, n_samples: int = 1600, sample_rate: int = 16000) -> np.ndarray:
        if z.ndim > 1:
            z = z[0]
        # Generate a simple waveform using z as harmonic coefficients
        t = np.linspace(0, n_samples / sample_rate, n_samples)
        wave = np.zeros(n_samples)
        for i, amp in enumerate(z[:8]):
            freq = 110 * (i + 1)  # harmonics of A2
            wave += amp * np.sin(2 * np.pi * freq * t)
        # Normalize
        wave = wave / (np.abs(wave).max() + 1e-8)
        return wave


# Head registry
HEADS: Dict[str, Head] = {
    "text": TextHead(),
    "image": ImageHead(),
    "audio": AudioHead(),
}


def decode_all_modalities(z: np.ndarray) -> Dict[str, object]:
    """Decode latent to all available modalities."""
    return {name: head.decode(z) for name, head in HEADS.items()}


if __name__ == "__main__":
    # Demo
    rng = np.random.default_rng(42)
    z = rng.normal(0, 1, 16).astype(np.float32)

    print("=== Text head ===")
    print(HEADS["text"].decode(z))
    print()
    print("=== Image head ===")
    print(HEADS["image"].decode(z))
    print()
    print("=== Audio head ===")
    audio = HEADS["audio"].decode(z, n_samples=1600)
    print(f"audio shape: {audio.shape}, range: [{audio.min():.3f}, {audio.max():.3f}]")
