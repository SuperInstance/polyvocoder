"""End-to-end polyvocoder pipeline.

encode_lore → JEV features → VAE.encode → z → VAE.decode → JEV-like output
                                                            ↓
                                                       output heads
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from jev_extractor import extract_jev_features, DEFAULT_QUESTIONS
from vae import VAE
from heads import HEADS, decode_all_modalities


def run_pipeline(
    text: str,
    latent_dim: int = 16,
    n_samples: int = 3,
    api_key: str = None,
    verbose: bool = True,
) -> dict:
    """Run the full polyvocoder pipeline on one lore.

    Steps:
    1. Extract JEV features for the lore
    2. Train (or load) a VAE on the single feature vector
    3. Sample n_samples latents from the VAE
    4. Decode each to all modalities

    Returns a dict with all outputs.
    """
    # Step 1: JEV features
    features = extract_jev_features(text, api_key=api_key)
    x = np.array(features, dtype=np.float32).reshape(1, -1)

    if verbose:
        print(f"JEV features ({len(features)}-dim):")
        for name, f in zip(DEFAULT_QUESTIONS.keys(), features):
            print(f"  {name}: {f:.3f}")

    # Step 2: VAE
    input_dim = len(features)
    vae = VAE(input_dim=input_dim, latent_dim=min(latent_dim, input_dim), hidden_dim=32)
    # Train briefly on this single example (warm start for the latent space)
    vae.fit(x, n_epochs=50, verbose=False)

    # Step 3: Sample latents
    mu, log_sigma = vae.encode(x)
    z_samples = []
    for i in range(n_samples):
        z = vae.reparameterize(mu, log_sigma)
        z_samples.append(z)

    if verbose:
        print(f"\nSampled {n_samples} latents (latent_dim={vae.latent_dim})")
        print(f"mu[0]: {mu[0]}")
        print(f"log_sigma[0]: {log_sigma[0]}")

    # Step 4: Decode each to all modalities
    decoded = []
    for i, z in enumerate(z_samples):
        outputs = decode_all_modalities(z)
        decoded.append(outputs)

    # Also decode the original mu
    mu_outputs = decode_all_modalities(mu)

    return {
        "features": features,
        "feature_names": list(DEFAULT_QUESTIONS.keys()),
        "mu": mu,
        "log_sigma": log_sigma,
        "z_samples": z_samples,
        "decoded_samples": decoded,
        "decoded_mu": mu_outputs,
        "vae": vae,
    }


if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else "The substrate walker is a 4D cell graph where cells are scars and the witness log accumulates."
    result = run_pipeline(text)

    print("\n" + "=" * 60)
    print("=== DECODED OUTPUTS ===")
    print("=" * 60)

    print("\n--- Sample 0 (text) ---")
    print(result["decoded_samples"][0]["text"])

    print("\n--- Sample 0 (image) ---")
    print(result["decoded_samples"][0]["image"])

    print("\n--- Sample 0 (audio shape) ---")
    audio = result["decoded_samples"][0]["audio"]
    print(f"shape={audio.shape}, range=[{audio.min():.3f}, {audio.max():.3f}]")

    print("\n--- mu (text) ---")
    print(result["decoded_mu"]["text"])
