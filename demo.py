"""Polyvocoder end-to-end demo.

This script shows the full polyvocoder pipeline:
1. Extract JEV features from canon lore
2. Fit tiny VAE on those features
3. Decode via 3 modality heads (text, image, audio)
4. Save outputs (lore text, ASCII art, WAV file)

Run:
    python3 demo.py
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "polyvocoder"))

from jev_extractor import extract_jev_features
from vae import VAE
from heads_v2 import TextHead, ImageHead, AudioHead
from save_audio import save_wav


SAMPLE_LORE = """The canon gate made itself heard. The substrate remembered itself
through scar tissue. The witness log accumulated, line by line, until the
city became the canon it had been waiting for."""


def main():
    print("=" * 60)
    print("Polyvocoder Demo")
    print("=" * 60)
    print()
    print(f"Input lore ({len(SAMPLE_LORE)} chars):")
    print(f"  {SAMPLE_LORE[:100]}...")
    print()

    # Step 1: Extract JEV features
    print("[1/4] Extracting JEV features via Typesafe.ai...")
    t0 = time.time()
    features = extract_jev_features(SAMPLE_LORE)
    print(f"      Done in {time.time() - t0:.1f}s")
    print(f"      Features (6-dim): {features}")
    print()

    # Step 2: Fit VAE
    print("[2/4] Fitting tiny VAE...")
    t0 = time.time()
    vae = VAE(input_dim=6, latent_dim=6, hidden_dim=16)
    vae.fit(features, n_epochs=30)
    print(f"      Done in {time.time() - t0:.1f}s")
    print()

    # Step 3: Sample and decode
    print("[3/4] Decoding via modality heads...")
    th = TextHead()
    ih = ImageHead()
    ah = AudioHead()

    n_samples = 3
    samples = []
    for i in range(n_samples):
        # Encode + reparameterize
        import numpy as np
        features_arr = np.array(features).reshape(1, -1)
        mu, log_sigma = vae.encode(features_arr)
        z = vae.reparameterize(mu, log_sigma).flatten()

        sample_text = th.decode(z)
        sample_image = ih.decode(z)
        sample_audio = ah.decode(z)
        samples.append({
            "latent": z.tolist(),
            "text": sample_text,
            "image": sample_image,
            "audio": sample_audio,
        })
        print(f"      Sample {i+1}: {sample_text[:80]}...")
    print()

    # Step 4: Save outputs
    print("[4/4] Saving outputs...")
    os.makedirs("demo_output", exist_ok=True)
    for i, s in enumerate(samples):
        # Save text
        with open(f"demo_output/sample_{i}.txt", "w") as f:
            f.write(s["text"])
        # Save image (ASCII art)
        with open(f"demo_output/sample_{i}.txt", "a") as f:
            f.write("\n\n" + s["image"] + "\n")
        # Save audio
        save_wav(np.array(s["audio"], dtype=np.float32), f"demo_output/sample_{i}.wav")
    print(f"      Saved to demo_output/sample_0.txt (and sample_1, sample_2)")
    print(f"      Saved to demo_output/sample_0.wav (and 1, 2)")
    print()

    # Save the JSON result
    import numpy as np
    result = {
        "input_lore": SAMPLE_LORE,
        "features": [float(x) for x in features],
        "decoded_samples": [
            {k: ([float(x) for x in v] if k == "audio" else v) for k, v in s.items()}
            for s in samples
        ],
    }
    with open("demo_output/result.json", "w") as f:
        json.dump(result, f, indent=2)

    print("✓ Demo complete. Output in demo_output/")
    print()
    print("Verify:")
    print("  cat demo_output/sample_0.txt   # text + ASCII art")
    print("  file demo_output/sample_0.wav  # audio metadata")


if __name__ == "__main__":
    main()
