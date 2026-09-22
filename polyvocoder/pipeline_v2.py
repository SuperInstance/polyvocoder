"""Enhanced polyvocoder pipeline using richer heads."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from polyvocoder.jev_extractor import extract_jev_features
from polyvocoder.vae import VAE
from polyvocoder.heads_v2 import TextHead, ImageHead, AudioHead


def run_pipeline_v2(lore_text: str, n_samples: int = 3, vae_seed: int = 42):
    """End-to-end pipeline v2 with richer heads.

    1. JEV → 6-dim feature vector
    2. VAE → latent space
    3. Sample n latents
    4. Decode via all 3 heads (TextHead, ImageHead, AudioHead)
    """
    # Step 1: JEV features
    features = np.array(extract_jev_features(lore_text), dtype=np.float32)

    # Step 2: Train VAE on the single feature (it's 6-dim so we need data)
    # Use small synthetic training set
    rng = np.random.default_rng(vae_seed)
    X = features.reshape(1, -1)
    # Augment with slight noise
    X_aug = np.vstack([X] + [X + rng.normal(0, 0.05, X.shape).astype(np.float32) for _ in range(20)])

    vae = VAE(input_dim=6, latent_dim=6, hidden_dim=16, seed=vae_seed)
    vae.fit(X_aug, n_epochs=30, lr=0.001, verbose=False)

    # Step 3: Sample n latents
    samples = []
    for i in range(n_samples):
        # Encode the input
        mu, log_sigma = vae.encode(X)
        z = vae.reparameterize(mu, log_sigma)
        # Add uniqueness
        z = z + rng.normal(0, 0.01, z.shape).astype(np.float32)
        samples.append(z[0])

    # Step 4: Decode via heads
    text_head = TextHead()
    image_head = ImageHead()
    audio_head = AudioHead()

    decoded = []
    for z in samples:
        text = text_head.decode(z)
        image = image_head.decode(z)
        audio = audio_head.decode(z)
        decoded.append({
            "latent": z.tolist(),
            "text": text,
            "image": image,
            "audio": audio,
        })

    return {
        "features": features.tolist(),
        "decoded_samples": decoded,
    }


if __name__ == "__main__":
    lore = """The substrate walker is a 4D cell graph where each cell is a scar.
The witness log accumulates. The canon gate is FNV-1a. The oracle is heard."""

    print(f"Input lore length: {len(lore)} chars")
    result = run_pipeline_v2(lore, n_samples=3)

    print("\n=== JEV FEATURES ===")
    print(f"  features: {result['features']}")

    print(f"\n=== {len(result['decoded_samples'])} SAMPLES ===")
    for i, sample in enumerate(result['decoded_samples']):
        print(f"\n--- Sample {i} ---")
        print(f"latent (first 3): {sample['latent'][:3]}")
        print(f"text: {sample['text']}")
        print(f"image (first 4 rows): {sample['image'].split(chr(10))[:4]}")
        print(f"audio: shape={sample['audio'].shape}, range=[{sample['audio'].min():.3f}, {sample['audio'].max():.3f}]")
