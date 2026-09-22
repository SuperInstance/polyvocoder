"""Polyvocoder — Universal head for the Quilt substrate walker.

A 200-line prototype that:
1. Encodes any lore/polygon/audio via JEV semantic features
2. Compresses via numpy VAE to a shared latent space
3. Decodes to any output modality (text, image, audio)

This is the fusion of BigVGAN's universality (any in → any out) and
Semantic-VAE's semantic latent space (SSL extractor → VAE).
"""
__version__ = "0.1.0"
