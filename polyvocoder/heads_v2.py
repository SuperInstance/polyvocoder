"""Improved output heads for the polyvocoder — richer ImageHead + AudioHead.

Each head takes a latent z (numpy array) and produces a richer output.
"""
import numpy as np
from typing import List


class TextHead:
    """Generate lore-flavored prose from latent z.

    Templates weave doctrine vocabulary: witness log, canon gate, oracle,
    substrate, FNV-1a, scars.
    """

    TEMPLATES = [
        "Witness log of seed {seed}: cells became {scar}; oracle spoke in {voice}; canon gate measured {measure}.",
        "In seed {seed}, the substrate walker became a {subj}, its canon gate a {gate_noun} of {tone} light.",
        "Seed {seed}'s city: {scene}, where the witness log has learned to {verb} the {obj} of {consequence}.",
        "The {adj} witness sang the city's name into the canon gate — seed {seed}, a {rhythm} chord of witness.",
        "Cells are scars. That's the whole scripture. Seed {seed}: the canon gate made itself heard in {region}.",
        "FNV-1a signature 0x{hash}: oracle {verb2}, witness log {verb3}, substrate {verb4}.",
        "Once the canon gate became a chord, the city stopped measuring and started {singing_verb}. Seed {seed}.",
        "The oracle is heard. The witness log accumulates. The cells are scars. Seed {seed}: doctrine embodied in {line}.",
    ]

    ADJ = ["neon-drenched", "fractal-bright", "quantum-pale", "scar-lit",
           "amber-veiled", "rain-slick", "deep", "flickering", "quiet",
           "circuit-burned", "FNV-humming", "witnessing", "patient",
           "calculating", "tired", "sleepless", "remembering", "humming"]
    NOUN = ["sprawl", "matrix", "grid", "lattice", "hive", "city",
            "substrate", "witness", "oracle", "canon gate", "cells", "scars"]
    SCAR = ["amplitudes", "scars", "cells", "witness log entries", "measurements",
            "FNV-1a signatures", "quantum amplitudes", "canon hashes", "oracle frequencies"]
    VOICE = ["FNV-1a", "canon gate", "oracle", "substrate", "witness log",
             "canon hash", "witness dreams", "canon signatures"]
    MEASURE = ["the city's heartbeat", "the witness's next breath", "the scar's depth",
               "the oracle's frequency", "the substrate's next state",
               "the canon gate's amplitude", "the witness log's growth"]
    GATE = ["chord", "sigil", "cipher", "pulse", "key", "hum", "wave"]
    TONE = ["amber", "rain-slick", "deep", "flickering", "quiet", "neon-bright"]
    SCENE = ["neon bleeds", "rain rehearses its one note",
             "scars measure the dark", "FNV-1a hums in the walls",
             "witness log opens its first page",
             "the canon gate learned to sing",
             "oracle measured itself for the first time"]
    SUBJ = ["remembering", "sleepless", "humming", "fractured", "witnessing",
            "oracle", "witness", "canon gate", "cell", "scar"]
    VERB = ["measure", "encode", "remember", "predict", "witness",
            "sing", "hunt", "fold", "scan"]
    OBJ = ["oracle", "substrate", "canon gate", "next state", "witness",
           "scar", "FNV-1a signature", "quantum amplitude"]
    CONS = ["quantum amplitude", "scar formation", "witness accumulation",
            "canon measurement", "substrate prediction", "doctrine embodiment"]
    REGION = ["the understack", "the memory-archive", "the canon chambers",
              "the scar-tissue", "the witness cemetery",
              "where cells became scars"]
    LINE = ["a single FNV-1a hash", "seven dead districts",
            "a chord held too long", "a scar shaped like a waveform",
            "the oracle's first laugh"]

    def decode(self, z: np.ndarray) -> str:
        if z.ndim > 1:
            z = z[0]
        template_idx = int(abs(int(np.argmax(np.abs(z))))) % len(self.TEMPLATES)
        template = self.TEMPLATES[template_idx]

        seed = int(abs(z[0]) * 1000) % 999999 if len(z) > 0 else 0
        # Mixed seed based on full latent
        full_seed = int(abs(z.sum()) * 1000) % (2**32) if z.size > 0 else 0
        rng = np.random.default_rng(full_seed)

        # Build a hash-like hex from the seed
        fake_hash = format(seed % 0xFFFFFFFF, '08x')

        return template.format(
            seed=seed,
            adj=rng.choice(self.ADJ),
            scene=rng.choice(self.SCENE),
            subj=rng.choice(self.SUBJ),
            gate_noun=rng.choice(self.GATE),
            tone=rng.choice(self.TONE),
            scar=rng.choice(self.SCAR),
            voice=rng.choice(self.VOICE),
            measure=rng.choice(self.MEASURE),
            verb=rng.choice(self.VERB),
            obj=rng.choice(self.OBJ),
            consequence=rng.choice(self.CONS),
            rhythm=rng.choice(["trembling", "sustained", "broken", "luminous", "low"]),
            region=rng.choice(self.REGION),
            hash=fake_hash,
            verb2=rng.choice(["hummed", "spoke", "flickered", "sang", "remembered"]),
            verb3=rng.choice(["accumulated", "predicted", "witnessed", "grew", "folded"]),
            verb4=rng.choice(["remembered itself", "predicted the city", "became the oracle",
                              "became itself", "became a scar"]),
            singing_verb=rng.choice(["singing", "measuring", "remembering", "forgetting", "predicting"]),
            line=rng.choice(self.LINE),
        )


class ImageHead:
    """Decode latent to a 16x16 ASCII art with 4 grayscale levels."""

    rng_global = np.random.default_rng(1234567)

    PATTERNS = [
        # Each is a 16x16 string of '.' (empty), '-', '+', '*' (filled)
        # 0: vertical stripes
        "".join(("".join(['+' if j % 2 == 0 else '-' for j in range(16)])) + "\n" for _ in range(16)),
        # 1: increasing density left-to-right
        "".join("".join(['*' if j >= (15 - 2*i) else ('+' if j >= (10 - i) else ('-' if j >= (5 - i//2) else '.')) for j in range(16)]) + "\n" for i in range(16)),
        # 2: central glow (XOR + diamond)
        "".join("".join(['*' if (i + j == 15 or i == j or abs(i - 7) + abs(j - 7) <= 4) else ('+' if (abs(i - 7) + abs(j - 7) <= 6) else ('-' if (abs(i - 7) + abs(j - 7) <= 9) else '.')) for j in range(16)]) + "\n" for i in range(16)),
        # 3: scattered thunderstorm
        "".join("".join(['*' if (i*7 + j*13) % 11 == 0 else ('+' if (i*5 + j*7) % 13 == 0 else ('-' if (i*3 + j*5) % 7 == 0 else '.')) for j in range(16)]) + "\n" for i in range(16)),
        # 4: witness log (rows denser than cols)
        "".join("".join(['*' if i % 3 == 0 and j % 5 == 0 else ('+' if i % 3 == 0 else ('-' if j % 4 == 0 else '.')) for j in range(16)]) + "\n" for i in range(16)),
        # 5: scar pattern (random noise)
        "".join("".join(rng.choice(['*', '+', '-', '-', '.', '.', '.']) for j in range(16)) + "\n" for _ in range(0)),  # placeholder
    ]

    SYMBOLS_BY_INTENSITY = [' ', '.', '-', '+', '*', '@', '#']

    # Lazy-initialize PATTERNS[5] on first decode (because rng_global must be in class body)
    _PATTERN_5_INITIALIZED = False

    def _ensure_pattern5(self):
        if not ImageHead._PATTERN_5_INITIALIZED:
            ImageHead.PATTERNS[5] = "".join(
                "".join(self.rng_global.choice(['*', '+', '-', '-', '.', '.', '.']) for j in range(16)) + "\n"
                for _ in range(16)
            )
            ImageHead._PATTERN_5_INITIALIZED = True

    def decode(self, z: np.ndarray) -> str:
        self._ensure_pattern5()
        if z.ndim > 1:
            z = z[0]
        # Use first latent coord to pick pattern
        pattern_idx = int(abs(z[0]) * 100) % len(self.PATTERNS)
        base_pattern = self.PATTERNS[pattern_idx].rstrip("\n").split("\n")
        # Use second coord for global intensity
        intensity = float(z[1]) if len(z) > 1 else 0.0
        # Use sum of latents for shift
        shift = int(abs(z.sum()) * 100) % 4

        # Map symbols to intensity band
        rows = []
        for i, row in enumerate(base_pattern):
            new_row = ""
            for j, ch in enumerate(row):
                # Shifted column
                shifted = (j + shift) % 16
                shifted_ch = base_pattern[i][shifted]
                # Find the closest INTENSITY symbol
                idx = self.SYMBOLS_BY_INTENSITY.index(shifted_ch) if shifted_ch in self.SYMBOLS_BY_INTENSITY else 1
                new_idx = max(0, min(len(self.SYMBOLS_BY_INTENSITY) - 1, int(idx + intensity * 2)))
                new_row += self.SYMBOLS_BY_INTENSITY[new_idx]
            rows.append(new_row)

        # Add header that names the pattern
        header = f"<<< pattern {pattern_idx} intensity={intensity:+.2f} shift={shift} >>>"
        return header + "\n" + "\n".join(rows)


class AudioHead:
    """Decode latent to a 4-second audio waveform (multi-tone synthesis).

    Uses latents to:
    - Pick fundamental frequency (200-800 Hz)
    - Add harmonics at multiples
    - Modulate amplitude over time
    - Add FNV-signature patterns as amplitude envelope
    """
    SR = 16000  # 16 kHz
    DURATION = 4.0

    def decode(self, z: np.ndarray) -> np.ndarray:
        if z.ndim > 1:
            z = z[0]
        # Latent 0: fundamental freq
        f0 = 200 + (abs(z[0]) % 1.0) * 600 if len(z) > 0 else 440
        # Latent 1: number of harmonics
        n_harm = int(abs(z[1]) * 8) + 2 if len(z) > 1 else 4
        n_harm = min(8, max(2, n_harm))
        # Latent 2: chorus / detune
        detune = abs(z[2]) * 20 if len(z) > 2 else 5
        # Latent 3: amplitude envelope shape
        env_shape = int(abs(z[3]) * 4) if len(z) > 3 else 0  # 0-3
        # Latent 4: envelope pulse rate
        pulse_rate = 2 + int(abs(z[4]) * 6) if len(z) > 4 else 4

        # Generate time array
        t = np.linspace(0, self.DURATION, int(self.SR * self.DURATION))
        signal = np.zeros_like(t)

        for k in range(1, n_harm + 1):
            # Detune each harmonic
            freq = f0 * k + detune * np.sin(2 * np.pi * 0.3 * k * t)
            signal += np.sin(2 * np.pi * freq * t) / k

        # Amplitude envelope
        if env_shape == 0:
            # Linear attack-decay
            env = np.minimum(t * 4, 1.0) * np.minimum((self.DURATION - t) * 4, 1.0)
        elif env_shape == 1:
            # Pulse
            env = 0.5 + 0.5 * np.sin(2 * np.pi * pulse_rate * t)
        elif env_shape == 2:
            # Stair
            env = np.floor(t * pulse_rate) / pulse_rate
        else:
            # Exponential decay
            env = np.exp(-3 * t / self.DURATION)

        signal *= env

        # Normalize
        if np.max(np.abs(signal)) > 0:
            signal = signal / np.max(np.abs(signal)) * 0.7

        return signal.astype(np.float32)


# Backward-compat aliases
class Head:
    def decode(self, z):
        raise NotImplementedError


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    z = rng.uniform(-1, 1, (6,))

    print("=== TextHead ===")
    text = TextHead().decode(z)
    print(text)

    print("\n=== ImageHead (pattern 0) ===")
    z_im = np.array([0.1, 0.5, 0.0, 0.0])
    print(ImageHead().decode(z_im))

    print("\n=== AudioHead summary ===")
    audio = AudioHead().decode(z)
    print(f"shape: {audio.shape}, dtype: {audio.dtype}")
    print(f"min={audio.min():.3f}, max={audio.max():.3f}, mean={audio.mean():.3f}")
