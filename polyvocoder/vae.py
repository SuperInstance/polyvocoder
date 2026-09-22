"""Numpy-only VAE for compressing JEV feature vectors."""
import numpy as np
from typing import Tuple, Dict


def _xavier_init(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    scale = np.sqrt(2.0 / (rows + cols))
    return (rng.normal(0, scale, size=(rows, cols)) * 0.1).astype(np.float32)


class VAE:
    """Variational autoencoder with Adam training via numerical gradients."""

    PARAMS = ["W_enc1", "b_enc1", "W_mu", "b_mu", "W_log_sigma", "b_log_sigma",
              "W_dec1", "b_dec1", "W_dec_out", "b_dec_out"]

    def __init__(self, input_dim: int, latent_dim: int = 4, hidden_dim: int = 16, seed: int = 42):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.rng = np.random.default_rng(seed)

        self.W_enc1 = _xavier_init(input_dim, hidden_dim, self.rng)
        self.b_enc1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W_mu = _xavier_init(hidden_dim, latent_dim, self.rng)
        self.b_mu = np.zeros(latent_dim, dtype=np.float32)
        self.W_log_sigma = _xavier_init(hidden_dim, latent_dim, self.rng)
        self.b_log_sigma = np.zeros(latent_dim, dtype=np.float32)
        self.W_dec1 = _xavier_init(latent_dim, hidden_dim, self.rng)
        self.b_dec1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W_dec_out = _xavier_init(hidden_dim, input_dim, self.rng)
        self.b_dec_out = np.zeros(input_dim, dtype=np.float32)

        self.m = {p: np.zeros_like(getattr(self, p)) for p in self.PARAMS}
        self.v = {p: np.zeros_like(getattr(self, p)) for p in self.PARAMS}

    def _relu(self, x):
        return np.maximum(0, x)

    def encode(self, x):
        h = self._relu(x @ self.W_enc1 + self.b_enc1)
        mu = h @ self.W_mu + self.b_mu
        log_sigma = np.clip(h @ self.W_log_sigma + self.b_log_sigma, -3, 3)
        return mu, log_sigma

    def reparameterize(self, mu, log_sigma):
        sigma = np.exp(log_sigma)
        eps = self.rng.normal(0, 1, size=mu.shape).astype(np.float32)
        return mu + sigma * eps

    def decode(self, z):
        h = self._relu(z @ self.W_dec1 + self.b_dec1)
        return np.clip(h @ self.W_dec_out + self.b_dec_out, -2, 2)

    def forward(self, x):
        mu, log_sigma = self.encode(x)
        z = self.reparameterize(mu, log_sigma)
        recon = self.decode(z)
        return recon, mu, log_sigma

    def loss(self, x, recon, mu, log_sigma, kl_weight=0.01):
        recon_loss = float(np.mean((x - recon) ** 2))
        kl_loss = float(-0.5 * np.mean(1 + log_sigma - mu ** 2 - np.exp(log_sigma)))
        return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss

    def _compute_grads(self, x, lr=0.001, kl_weight=0.01):
        eps = 1e-3
        recon, mu, log_sigma = self.forward(x)
        total, _, _ = self.loss(x, recon, mu, log_sigma, kl_weight)

        grads = {p: np.zeros_like(getattr(self, p)) for p in self.PARAMS}
        for pname in self.PARAMS:
            p = getattr(self, pname)
            if p.size > 30:
                idx = self.rng.choice(p.size, size=10, replace=False)
            else:
                idx = np.arange(p.size)
            flat_p = p.flatten()
            for fi in idx:
                old = flat_p[fi]
                flat_p[fi] = old + eps
                setattr(self, pname, flat_p.reshape(p.shape))
                r2, m2, ls2 = self.forward(x)
                l2, _, _ = self.loss(x, r2, m2, ls2, kl_weight)
                grads[pname].flat[fi] = (l2 - total) / eps
                flat_p[fi] = old
            setattr(self, pname, flat_p.reshape(p.shape))
        return grads

    def fit(self, data, n_epochs=30, lr=0.001, kl_weight=0.01, verbose=False):
        losses = []
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        t = 0
        for epoch in range(n_epochs):
            recon, mu, log_sigma = self.forward(data)
            total, recon_l, kl_l = self.loss(data, recon, mu, log_sigma, kl_weight)
            losses.append(total)
            if verbose and epoch % 5 == 0:
                print(f"epoch {epoch}: total={total:.4f} recon={recon_l:.4f} kl={kl_l:.4f}")
            grads = self._compute_grads(data, lr=lr, kl_weight=kl_weight)
            t += 1
            for pname in self.PARAMS:
                g = grads[pname]
                self.m[pname] = beta1 * self.m[pname] + (1 - beta1) * g
                self.v[pname] = beta2 * self.v[pname] + (1 - beta2) * (g ** 2)
                m_hat = self.m[pname] / (1 - beta1 ** t)
                v_hat = self.v[pname] / (1 - beta2 ** t)
                update = lr * m_hat / (np.sqrt(v_hat) + eps)
                setattr(self, pname, getattr(self, pname) - update)
        return losses


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (100, 8)).astype(np.float32)
    vae = VAE(input_dim=8, latent_dim=4, hidden_dim=16)
    losses = vae.fit(X, n_epochs=20, verbose=True)
    print(f"Final: {losses[-1]:.4f}")
    x_test = rng.uniform(0, 1, (5, 8)).astype(np.float32)
    recon, mu, log_sigma = vae.forward(x_test)
    print(f"Recon err: {float(np.mean((x_test - recon) ** 2)):.4f}")
