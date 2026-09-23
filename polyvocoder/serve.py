"""Polyvocoder HTTP server — stdlib-only implementation.

Exposes POST /v1/pipeline that accepts a LoreInput and returns a PipelineResult.
Uses Python's stdlib http.server (no Flask dependency).
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(__file__))

from jev_extractor import extract_jev_features as extract_features
from vae import VAE
from heads_v2 import TextHead, ImageHead, AudioHead

PORT = int(os.environ.get("PORT", 8000))

# Initialize the polyvocoder components (lazy)
vae = None
text_head = None
image_head = None
audio_head = None


def get_components():
    global vae, text_head, image_head, audio_head
    if vae is None:
        vae = VAE(input_dim=6, latent_dim=6, hidden_dim=16)
    if text_head is None:
        text_head = TextHead()
    if image_head is None:
        image_head = ImageHead()
    if audio_head is None:
        audio_head = AudioHead()
    return vae, text_head, image_head, audio_head


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_OPTIONS(self):
        """CORS preflight."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/canary":
            from canary import canary_hash
            return self._send_json({"canary": f"0x{canary_hash():016x}"})
        if self.path == "/health":
            return self._send_json({"status": "ok"})
        if self.path.startswith("/v1/"):
            return self._send_json({"error": "GET not supported on this endpoint"}, 405)
        return self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/v1/pipeline":
            return self._handle_pipeline()
        if self.path == "/v1/features":
            return self._handle_features()
        return self._send_json({"error": "not found"}, 404)

    def _handle_features(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            text = data.get('text', '')
            if not text:
                return self._send_json({"error": "Missing 'text'"}, 400)
            fv = extract_features(text)
            keys = ["canon_worthy", "distinct_voice", "doctrine_anchor", "voice_fit", "novelty", "density"]
            return self._send_json(dict(zip(keys, fv)))
        except Exception as e:
            return self._send_json({"error": str(e)}, 500)

    def _handle_pipeline(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            text = data.get('text', '')
            if not text:
                return self._send_json({"error": "Missing 'text'"}, 400)
            n_samples = data.get('n_samples', 3)
            seed = data.get('seed', 42)

            features = extract_features(text)
            v, th, ih, ah = get_components()
            v.fit(features)
            import numpy as np
            features_arr = np.array(features).reshape(1, -1)
            mu, log_sigma = v.encode(features_arr)
            # Convert to numpy arrays
            import numpy as np
            if not isinstance(mu, np.ndarray):
                mu = np.array(mu)
            if not isinstance(log_sigma, np.ndarray):
                log_sigma = np.array(log_sigma)
            latents = []
            for i in range(n_samples):
                z = v.reparameterize(mu, log_sigma)
                z = np.array(z).flatten()
                latents.append([float(x) for x in z])

            decoded_samples = []
            for latent in latents:
                latent_arr = np.array(latent)
                sample_text = th.decode(latent_arr)
                sample_image = ih.decode(latent_arr)
                sample_audio = ah.decode(latent_arr)
                sample_audio = np.array(sample_audio).flatten().tolist()
                decoded_samples.append({
                    "latent": list(latent),
                    "text": sample_text,
                    "image": sample_image,
                    "audio": list(sample_audio),
                })

            return self._send_json({
                "features": features,
                "decoded_samples": decoded_samples,
            })
        except Exception as e:
            return self._send_json({"error": str(e)}, 500)

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass


def run():
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Polyvocoder server listening on http://0.0.0.0:{PORT}")
    print(f"Try: curl http://localhost:{PORT}/canary")
    server.serve_forever()


if __name__ == "__main__":
    run()
