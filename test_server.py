"""Test the polyvocoder HTTP server."""
import json
import time
import sys

import requests

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"


def test_canary():
    """Verify the canary endpoint."""
    r = requests.get(f"{URL}/canary")
    print(f"GET /canary: {r.json()}")
    assert r.status_code == 200
    assert "0x024a555471370b18d" in r.json()['canary']


def test_features():
    """Verify the features endpoint."""
    r = requests.post(f"{URL}/v1/features", json={"text": "The cells are scars."})
    print(f"POST /v1/features: {r.json()}")
    assert r.status_code == 200
    assert 'canon_worthy' in r.json()


def test_pipeline():
    """Verify the pipeline endpoint."""
    r = requests.post(f"{URL}/v1/pipeline", json={
        "text": "The canon gate made itself heard.",
        "n_samples": 2,
    })
    print(f"POST /v1/pipeline: {r.status_code}, keys={list(r.json().keys())}")
    assert r.status_code == 200
    assert 'features' in r.json()
    assert 'decoded_samples' in r.json()


if __name__ == "__main__":
    test_canary()
    test_features()
    test_pipeline()
    print("\n✓ All tests passed!")
